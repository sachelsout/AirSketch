"""
scripts/export_onnx.py

Exports the best AirSketch ST-Transformer checkpoint to ONNX format,
validates numerical parity against the PyTorch model, benchmarks
CPU inference latency, and writes a machine-readable results JSON.

Steps:
    1. Load best_model.pt checkpoint
    2. Export to ONNX with opset 17
    3. Run ONNX graph optimizations
    4. Validate parity: max |onnx_out - torch_out| < 1e-5
    5. Benchmark latency over 200 warmup + 500 timed runs
    6. Write checkpoints/export_results.json with all measurements
    7. Exit with code 1 if latency target or parity check fails

Usage:
    python scripts/export_onnx.py \
        --checkpoint checkpoints/best_model.pt \
        --output     checkpoints/best_model.onnx \
        --config     configs/default.yaml

    # Benchmark on a specific number of runs
    python scripts/export_onnx.py \
        --checkpoint checkpoints/best_model.pt \
        --output     checkpoints/best_model.onnx \
        --n-warmup   200 \
        --n-bench    500
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

from src.model import build_model, STTransformer


# -- Constants -----------------------------------------------------------------

LATENCY_TARGET_MS = 50.0  # hard requirement from project proposal
PARITY_TOLERANCE = 1e-5  # max absolute difference allowed
OPSET_VERSION = 17
INPUT_NAME = "sequence"
OUTPUT_NAMES = ["pred_xy", "gesture_logits"]
IMAGE_SIZE = 224  # used to report MPJPE-equivalent latency context


# -- Model loading -------------------------------------------------------------


def load_pytorch_model(
    checkpoint_path: Path,
    config_path: Path,
    device: str = "cpu",
) -> tuple[STTransformer, dict]:
    """
    Load the trained ST-Transformer from a checkpoint.

    Loads the config embedded in the checkpoint (saved by train.py).
    Falls back to the provided config_path if the checkpoint does not
    contain a config block (older checkpoint format).

    Returns:
        (model, config_dict)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Prefer config stored in the checkpoint for reproducibility
    if "config" in ckpt and ckpt["config"]:
        config = ckpt["config"]
        print("  Config loaded from checkpoint.")
    else:
        print(f"  Config not found in checkpoint -- loading from {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)

    model = build_model(config, device=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    best_mpjpe = ckpt.get("best_mpjpe", float("nan"))
    epoch = ckpt.get("epoch", "unknown")
    print(f"  Epoch: {epoch}  |  Best val MPJPE: {best_mpjpe:.4f} px")

    return model, config


# -- ONNX export ---------------------------------------------------------------


def export_to_onnx(
    model: STTransformer,
    output_path: Path,
    opset: int = OPSET_VERSION,
) -> None:
    """
    Export the PyTorch model to ONNX format.

    Uses a fixed batch size of 1 (inference only processes one frame
    window at a time in the real-time loop). The batch dimension is
    exported as dynamic so the ONNX model can be validated with
    multiple batch sizes during parity testing.

    Args:
        model:       STTransformer in eval mode, on CPU.
        output_path: Where to write the .onnx file.
        opset:       ONNX opset version.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Representative input -- same shape as real-time inference
    dummy_input = torch.zeros(1, 16, 42, dtype=torch.float32)

    print(f"\nExporting to ONNX (opset {opset}) ...")
    print(f"  Input shape:  {tuple(dummy_input.shape)}")
    print(f"  Output path:  {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,  # folds constant subgraphs at export time
        input_names=[INPUT_NAME],
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            INPUT_NAME: {0: "batch_size"},
            OUTPUT_NAMES[0]: {0: "batch_size"},
            OUTPUT_NAMES[1]: {0: "batch_size"},
        },
    )

    size_mb = output_path.stat().st_size / 1e6
    print(f"  Export complete.  File size: {size_mb:.1f} MB")


# -- ONNX graph optimization ---------------------------------------------------


def optimize_onnx(input_path: Path, output_path: Path) -> Path:
    """
    Run ONNX Runtime's graph optimization pass on the exported model.

    ORT applies level-3 (all) optimizations: operator fusion, redundant
    node elimination, and constant propagation. The optimized model is
    saved to a separate file -- the unoptimized version is kept for
    comparison.

    Args:
        input_path:  Path to the raw exported .onnx file.
        output_path: Path for the optimized .onnx file.

    Returns:
        output_path (for chaining).
    """
    print("\nRunning ONNX Runtime graph optimizations ...")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = str(output_path)

    # Creating a session with optimized_model_filepath triggers the
    # optimization pass and writes the result to disk
    _ = ort.InferenceSession(
        str(input_path),
        sess_options,
        providers=["CPUExecutionProvider"],
    )

    size_before = input_path.stat().st_size / 1e6
    size_after = output_path.stat().st_size / 1e6
    print(f"  Before: {size_before:.1f} MB  -->  After: {size_after:.1f} MB")

    return output_path


# -- Parity validation ---------------------------------------------------------


def validate_parity(
    model: STTransformer,
    onnx_path: Path,
    n_samples: int = 100,
    tolerance: float = PARITY_TOLERANCE,
) -> dict[str, float]:
    """
    Compare ONNX Runtime outputs against PyTorch outputs on random inputs.

    Tests n_samples random inputs and reports the maximum absolute
    difference across all outputs and all samples. Both pred_xy and
    gesture_logits are checked independently.

    Args:
        model:     PyTorch model in eval mode on CPU.
        onnx_path: Path to the ONNX model file to validate.
        n_samples: Number of random test inputs.
        tolerance: Maximum allowed absolute difference.

    Returns:
        Dict with keys max_diff_xy, max_diff_gesture, passed (bool).

    Raises:
        AssertionError if max_diff exceeds tolerance.
    """
    print(f"\nValidating numerical parity ({n_samples} random inputs) ...")

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    max_diff_xy = 0.0
    max_diff_gesture = 0.0
    rng = np.random.default_rng(42)

    for i in range(n_samples):
        # Random input in [0, 1] -- same range as real landmark data
        x_np = rng.random((1, 16, 42), dtype=np.float32)
        x_torch = torch.from_numpy(x_np)

        # PyTorch forward pass
        with torch.no_grad():
            torch_out = model(x_torch)
        torch_xy = torch_out["pred_xy"].numpy()
        torch_ges = torch_out["gesture_logits"].numpy()

        # ONNX Runtime forward pass
        ort_outputs = sess.run(
            OUTPUT_NAMES,
            {INPUT_NAME: x_np},
        )
        ort_xy = ort_outputs[0]
        ort_ges = ort_outputs[1]

        # Per-sample max absolute difference
        diff_xy = float(np.max(np.abs(torch_xy - ort_xy)))
        diff_gesture = float(np.max(np.abs(torch_ges - ort_ges)))

        max_diff_xy = max(max_diff_xy, diff_xy)
        max_diff_gesture = max(max_diff_gesture, diff_gesture)

        if diff_xy > tolerance or diff_gesture > tolerance:
            print(
                f"  FAIL at sample {i}: "
                f"diff_xy={diff_xy:.2e}  diff_gesture={diff_gesture:.2e}"
            )

    passed = max_diff_xy <= tolerance and max_diff_gesture <= tolerance
    status = "PASS" if passed else "FAIL"

    print(
        f"  pred_xy       max diff: {max_diff_xy:.2e}  "
        f"(tolerance: {tolerance:.0e})  [{status}]"
    )
    print(
        f"  gesture_logit max diff: {max_diff_gesture:.2e}  "
        f"(tolerance: {tolerance:.0e})  [{status}]"
    )

    if not passed:
        raise AssertionError(
            f"Parity check FAILED. "
            f"max_diff_xy={max_diff_xy:.2e}, "
            f"max_diff_gesture={max_diff_gesture:.2e}, "
            f"tolerance={tolerance:.0e}. "
            f"The ONNX export may have introduced unsupported operations. "
            f"Check for custom layers or ops not covered by opset {OPSET_VERSION}."
        )

    print("  Parity check PASSED.")
    return {
        "max_diff_xy": max_diff_xy,
        "max_diff_gesture": max_diff_gesture,
        "tolerance": tolerance,
        "n_samples": n_samples,
        "passed": passed,
    }


# -- Latency benchmark ---------------------------------------------------------


def benchmark_latency(
    onnx_path: Path,
    n_warmup: int = 200,
    n_bench: int = 500,
) -> dict[str, float]:
    """
    Benchmark end-to-end ONNX Runtime inference latency on CPU.

    Runs n_warmup passes to warm up the session cache and JIT compilation,
    then times n_bench passes and reports statistics.

    All measurements use time.perf_counter() for sub-millisecond resolution.
    Results are in milliseconds.

    The latency measured here corresponds to:
        MediaPipe already ran (landmark extraction is separate)
        Window assembled (16 frames already in the buffer)
        ONNX model runs on the 16-frame window
        pred_xy and gesture_logits returned

    End-to-end latency including MediaPipe is benchmarked separately
    in issue #13.

    Args:
        onnx_path: Path to the optimized ONNX model.
        n_warmup:  Number of warmup passes (not timed).
        n_bench:   Number of timed passes.

    Returns:
        Dict with mean_ms, median_ms, p95_ms, p99_ms, min_ms, max_ms,
              std_ms, target_ms, passed (bool).
    """
    print("\nBenchmarking CPU inference latency ...")
    print(f"  Model:   {onnx_path}")
    print(f"  Warmup:  {n_warmup} runs")
    print(f"  Timed:   {n_bench} runs")
    print(f"  Target:  < {LATENCY_TARGET_MS:.0f} ms per frame")

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    # Fixed input -- same value every run to isolate model latency
    # from input generation time
    x = np.zeros((1, 16, 42), dtype=np.float32)

    # Warmup -- allow ORT to cache memory allocations and optimize dispatch
    print("  Warming up ...", end=" ", flush=True)
    for _ in range(n_warmup):
        sess.run(OUTPUT_NAMES, {INPUT_NAME: x})
    print("done.")

    # Timed runs
    latencies_ms = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        sess.run(OUTPUT_NAMES, {INPUT_NAME: x})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)

    mean_ms = float(np.mean(latencies_ms))
    median_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))
    std_ms = float(np.std(latencies_ms))

    # Pass/fail uses P95 -- a real-time system must meet the target
    # consistently, not just on average. If P95 > 50 ms, 1 in 20
    # frames will miss the deadline, which causes visible jitter.
    passed = p95_ms < LATENCY_TARGET_MS

    print(f"\n  Latency statistics ({n_bench} runs):")
    print(f"    Mean:    {mean_ms:6.2f} ms")
    print(f"    Median:  {median_ms:6.2f} ms")
    print(f"    P95:     {p95_ms:6.2f} ms  {'[PASS]' if passed else '[FAIL]'}")
    print(f"    P99:     {p99_ms:6.2f} ms")
    print(f"    Min:     {min_ms:6.2f} ms")
    print(f"    Max:     {max_ms:6.2f} ms")
    print(f"    Std:     {std_ms:6.2f} ms")

    if not passed:
        print(
            f"\n  LATENCY TARGET FAILED: P95 ({p95_ms:.2f} ms) > {LATENCY_TARGET_MS:.0f} ms"
        )
        print("  See troubleshooting section in issue #12 for remedies.")
    else:
        print(
            f"\n  Latency target MET: P95 {p95_ms:.2f} ms < {LATENCY_TARGET_MS:.0f} ms"
        )

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "target_ms": LATENCY_TARGET_MS,
        "n_warmup": n_warmup,
        "n_bench": n_bench,
        "passed": passed,
    }


# -- PyTorch CPU baseline benchmark --------------------------------------------


def benchmark_pytorch_baseline(
    model: STTransformer,
    n_bench: int = 200,
) -> dict[str, float]:
    """
    Benchmark PyTorch CPU inference as a baseline for the ONNX speedup ratio.

    Uses the same protocol as benchmark_latency() but runs the PyTorch
    model directly instead of ONNX Runtime. This gives the speedup ratio:
        speedup = pytorch_mean_ms / onnx_mean_ms

    Args:
        model:   STTransformer in eval mode on CPU.
        n_bench: Number of timed passes (fewer than ONNX since it is slower).

    Returns:
        Dict with mean_ms and speedup_vs_onnx (filled in by caller).
    """
    print(f"\nBenchmarking PyTorch CPU baseline ({n_bench} runs) ...")

    x = torch.zeros(1, 16, 42, dtype=torch.float32)

    # Warmup
    for _ in range(50):
        with torch.no_grad():
            model(x)

    latencies_ms = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(latencies_ms))
    median_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))

    print(
        f"  PyTorch CPU:  mean={mean_ms:.2f} ms  "
        f"median={median_ms:.2f} ms  P95={p95_ms:.2f} ms"
    )

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
    }


# -- ONNX model validation (graph structure) -----------------------------------


def validate_onnx_graph(onnx_path: Path) -> None:
    """
    Run ONNX's built-in graph checker to verify the exported model is
    well-formed (no unsupported ops, no broken references, correct shapes).

    Raises onnx.checker.ValidationError on failure.
    """
    print("\nValidating ONNX graph structure ...")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print(
        f"  Graph valid.  Nodes: {len(model.graph.node)}  "
        f"Inputs: {len(model.graph.input)}  "
        f"Outputs: {len(model.graph.output)}"
    )

    # Print input and output shapes
    for inp in model.graph.input:
        shape = [
            d.dim_value if d.dim_value > 0 else d.dim_param
            for d in inp.type.tensor_type.shape.dim
        ]
        print(f"  Input  '{inp.name}': {shape}")
    for out in model.graph.output:
        shape = [
            d.dim_value if d.dim_value > 0 else d.dim_param
            for d in out.type.tensor_type.shape.dim
        ]
        print(f"  Output '{out.name}': {shape}")


# -- Main ----------------------------------------------------------------------


def main(
    checkpoint_path: str,
    output_path: str,
    config_path: str,
    n_warmup: int,
    n_bench: int,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    config_path = Path(config_path)
    optimized_path = output_path.with_suffix(".optimized.onnx")
    results_path = output_path.parent / "export_results.json"

    # -- Load model ------------------------------------------------------------
    model, config = load_pytorch_model(checkpoint_path, config_path)
    model.eval()

    # -- Export ----------------------------------------------------------------
    export_to_onnx(model, output_path)

    # -- Validate graph structure ----------------------------------------------
    validate_onnx_graph(output_path)

    # -- Optimize --------------------------------------------------------------
    optimize_onnx(output_path, optimized_path)
    validate_onnx_graph(optimized_path)  # re-validate after optimization

    # -- Parity check ----------------------------------------------------------
    parity_results = validate_parity(model, optimized_path)

    # -- Latency benchmark -----------------------------------------------------
    latency_results = benchmark_latency(optimized_path, n_warmup, n_bench)

    # -- PyTorch baseline ------------------------------------------------------
    pytorch_results = benchmark_pytorch_baseline(model)
    speedup = pytorch_results["mean_ms"] / max(latency_results["mean_ms"], 0.001)
    pytorch_results["speedup_vs_onnx"] = round(speedup, 2)

    print(f"\n  ONNX Runtime speedup vs PyTorch CPU: {speedup:.1f}x")

    # -- Write results JSON ----------------------------------------------------
    import platform

    results = {
        "checkpoint": str(checkpoint_path),
        "onnx_path": str(output_path),
        "optimized_path": str(optimized_path),
        "opset_version": OPSET_VERSION,
        "onnxruntime_version": ort.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "parity": parity_results,
        "latency_onnx": latency_results,
        "latency_pytorch": pytorch_results,
        "model_config": {
            "hidden_dim": config.get("model", {}).get("hidden_dim"),
            "num_layers": config.get("model", {}).get("num_layers"),
            "num_heads": config.get("model", {}).get("num_heads"),
        },
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results written: {results_path}")

    # -- Final pass/fail summary -----------------------------------------------
    print(f"\n{'--'*30}")
    parity_ok = parity_results["passed"]
    latency_ok = latency_results["passed"]

    print(f"  Parity check:    {'PASS' if parity_ok  else 'FAIL'}")
    print(
        f"  Latency target:  {'PASS' if latency_ok else 'FAIL'}  "
        f"(P95 = {latency_results['p95_ms']:.1f} ms, "
        f"target < {LATENCY_TARGET_MS:.0f} ms)"
    )
    print(f"{'--'*30}")

    if not parity_ok or not latency_ok:
        print("\nExport FAILED. See troubleshooting section in issue #12.")
        raise SystemExit(1)

    print("\nExport PASSED. ONNX model ready for issue #13.")
    print(f"  Use: {optimized_path}")


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Export AirSketch ST-Transformer to ONNX for CPU inference."
    )
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--output", default="checkpoints/best_model.onnx")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n-warmup", type=int, default=200)
    p.add_argument("--n-bench", type=int, default=500)
    args = p.parse_args()
    main(args.checkpoint, args.output, args.config, args.n_warmup, args.n_bench)
