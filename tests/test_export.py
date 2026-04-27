"""
tests/test_export.py

Unit tests for ONNX export and validation functions.
Uses a tiny model (hidden_dim=32, num_layers=1) for speed.
All tests run on CPU without requiring a real trained checkpoint.
"""

import numpy as np
import onnx
import onnxruntime as ort

from src.model import STTransformer, STTransformerConfig
from scripts.export_onnx import (
    export_to_onnx,
    validate_onnx_graph,
    validate_parity,
    benchmark_latency,
    PARITY_TOLERANCE,
    INPUT_NAME,
    OUTPUT_NAMES,
)


# -- Fixture -------------------------------------------------------------------


def small_model() -> STTransformer:
    cfg = STTransformerConfig(hidden_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    m = STTransformer(cfg)
    m.eval()
    return m


# -- Export tests --------------------------------------------------------------


class TestExportToOnnx:

    def test_file_created(self, tmp_path):
        model = small_model()
        out = tmp_path / "model.onnx"
        export_to_onnx(model, out)
        assert out.exists(), "ONNX file was not created"
        assert out.stat().st_size > 0, "ONNX file is empty"

    def test_onnx_graph_valid(self, tmp_path):
        model = small_model()
        out = tmp_path / "model.onnx"
        export_to_onnx(model, out)
        validate_onnx_graph(out)  # raises on invalid graph

    def test_correct_input_output_names(self, tmp_path):
        model = small_model()
        out = tmp_path / "model.onnx"
        export_to_onnx(model, out)
        onnx_mdl = onnx.load(str(out))
        input_names = [i.name for i in onnx_mdl.graph.input]
        output_names = [o.name for o in onnx_mdl.graph.output]
        assert INPUT_NAME in input_names, f"'{INPUT_NAME}' not in {input_names}"
        assert (
            OUTPUT_NAMES[0] in output_names
        ), f"'{OUTPUT_NAMES[0]}' not in {output_names}"
        assert (
            OUTPUT_NAMES[1] in output_names
        ), f"'{OUTPUT_NAMES[1]}' not in {output_names}"

    def test_ort_session_loads(self, tmp_path):
        """ONNX Runtime must be able to load the exported model."""
        model = small_model()
        out = tmp_path / "model.onnx"
        export_to_onnx(model, out)
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        assert sess is not None

    def test_ort_produces_correct_output_shapes(self, tmp_path):
        model = small_model()
        out = tmp_path / "model.onnx"
        export_to_onnx(model, out)
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        x = np.zeros((1, 16, 42), dtype=np.float32)
        outputs = sess.run(OUTPUT_NAMES, {INPUT_NAME: x})
        assert outputs[0].shape == (1, 2), f"pred_xy shape: {outputs[0].shape}"
        assert outputs[1].shape == (1, 2), f"gesture_logits shape: {outputs[1].shape}"

    def test_pred_xy_in_unit_range(self, tmp_path):
        """pred_xy uses Sigmoid -- output must be in [0, 1]."""
        model = small_model()
        out = tmp_path / "model.onnx"
        export_to_onnx(model, out)
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = rng.random((1, 16, 42), dtype=np.float32)
            outs = sess.run(OUTPUT_NAMES, {INPUT_NAME: x})
            xy = outs[0]
            assert xy.min() >= 0.0, f"pred_xy below 0: {xy.min()}"
            assert xy.max() <= 1.0, f"pred_xy above 1: {xy.max()}"


# -- Parity tests --------------------------------------------------------------


class TestValidateParity:

    def test_parity_passes_on_fresh_model(self, tmp_path):
        """A freshly exported model must pass parity without any fine-tuning."""
        model = small_model()
        out = tmp_path / "model.onnx"
        export_to_onnx(model, out)
        result = validate_parity(model, out, n_samples=20)
        assert result["passed"], (
            f"Parity failed: max_diff_xy={result['max_diff_xy']:.2e}, "
            f"max_diff_gesture={result['max_diff_gesture']:.2e}"
        )

    def test_parity_result_contains_required_keys(self, tmp_path):
        model = small_model()
        out = tmp_path / "model.onnx"
        export_to_onnx(model, out)
        result = validate_parity(model, out, n_samples=5)
        for key in [
            "max_diff_xy",
            "max_diff_gesture",
            "tolerance",
            "n_samples",
            "passed",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_parity_max_diff_below_tolerance(self, tmp_path):
        model = small_model()
        out = tmp_path / "model.onnx"
        export_to_onnx(model, out)
        result = validate_parity(model, out, n_samples=20, tolerance=PARITY_TOLERANCE)
        assert result["max_diff_xy"] < PARITY_TOLERANCE
        assert result["max_diff_gesture"] < PARITY_TOLERANCE


# -- Latency tests -------------------------------------------------------------


class TestBenchmarkLatency:

    def test_latency_result_contains_required_keys(self, tmp_path):
        model = small_model()
        out = tmp_path / "model.onnx"
        export_to_onnx(model, out)
        result = benchmark_latency(out, n_warmup=5, n_bench=20)
        for key in [
            "mean_ms",
            "median_ms",
            "p95_ms",
            "p99_ms",
            "min_ms",
            "max_ms",
            "std_ms",
            "target_ms",
            "n_warmup",
            "n_bench",
            "passed",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_latency_values_are_positive(self, tmp_path):
        model = small_model()
        out = tmp_path / "model.onnx"
        export_to_onnx(model, out)
        result = benchmark_latency(out, n_warmup=5, n_bench=20)
        assert result["mean_ms"] > 0
        assert result["median_ms"] > 0
        assert result["p95_ms"] > 0
        assert result["min_ms"] > 0

    def test_small_model_meets_latency_target(self, tmp_path):
        """
        A hidden_dim=32 model should be fast enough to meet the 50 ms target
        on any development machine. If this fails, the ORT setup is broken.
        """
        model = small_model()
        out = tmp_path / "model.onnx"
        export_to_onnx(model, out)
        result = benchmark_latency(out, n_warmup=10, n_bench=50)
        assert result["passed"], (
            f"Small model missed latency target: P95={result['p95_ms']:.1f} ms. "
            f"ONNX Runtime may not be configured correctly."
        )

    def test_n_bench_runs_are_timed(self, tmp_path):
        """Exactly n_bench latency measurements must be recorded."""
        model = small_model()
        out = tmp_path / "model.onnx"
        export_to_onnx(model, out)
        result = benchmark_latency(out, n_warmup=2, n_bench=15)
        assert result["n_bench"] == 15
