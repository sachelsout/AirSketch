"""
tests/test_model.py

Unit tests for STTransformer architecture.

All tests use small configs (hidden_dim=32, num_layers=2) for speed.
They verify output shapes, dtypes, value ranges, and gradient flow —
not model quality. The model is randomly initialized throughout.
"""

import pytest
import torch

from src.model import (
    STTransformer,
    STTransformerConfig,
    AirSketchLoss,
    FeedForward,
    SpatialSelfAttention,
    TemporalSelfAttention,
    STBlock,
    RegressionHead,
    GestureHead,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

BATCH = 4
T = 16
J = 21
D_SMALL = 32  # small hidden dim for fast tests


def small_config(**overrides) -> STTransformerConfig:
    """Return a minimal config for fast unit tests."""
    defaults = dict(hidden_dim=D_SMALL, num_layers=2, num_heads=4, dropout=0.0)
    defaults.update(overrides)
    return STTransformerConfig(**defaults)


def random_batch(B: int = BATCH, T: int = T) -> torch.Tensor:
    """Return a random (B, T, 42) float32 input tensor in [0, 1]."""
    return torch.rand(B, T, 42)


# ── Sub-module shape tests ─────────────────────────────────────────────────────


class TestSubModules:

    def test_feedforward_shape(self):
        ffn = FeedForward(d_model=D_SMALL, dropout=0.0)
        x = torch.rand(BATCH, T, J, D_SMALL)
        out = ffn(x)
        assert out.shape == x.shape

    def test_spatial_attention_shape(self):
        attn = SpatialSelfAttention(D_SMALL, num_heads=4, dropout=0.0)
        x = torch.rand(BATCH, T, J, D_SMALL)
        out = attn(x)
        assert out.shape == (BATCH, T, J, D_SMALL)

    def test_temporal_attention_shape(self):
        attn = TemporalSelfAttention(D_SMALL, num_heads=4, dropout=0.0)
        x = torch.rand(BATCH, T, J, D_SMALL)
        out = attn(x)
        assert out.shape == (BATCH, T, J, D_SMALL)

    def test_stblock_shape(self):
        block = STBlock(D_SMALL, num_heads=4, dropout=0.0)
        x = torch.rand(BATCH, T, J, D_SMALL)
        out = block(x)
        assert out.shape == (BATCH, T, J, D_SMALL)

    def test_regression_head_shape(self):
        head = RegressionHead(D_SMALL, dropout=0.0)
        x = torch.rand(BATCH, D_SMALL)
        out = head(x)
        assert out.shape == (BATCH, 2)

    def test_gesture_head_shape(self):
        head = GestureHead(D_SMALL)
        x = torch.rand(BATCH, D_SMALL)
        out = head(x)
        assert out.shape == (BATCH, 2)

    def test_regression_head_output_range(self):
        """Regression head uses Sigmoid — output must be in [0, 1]."""
        head = RegressionHead(D_SMALL, dropout=0.0)
        x = torch.randn(BATCH, D_SMALL) * 10  # large inputs to stress sigmoid
        out = head(x)
        assert out.min() >= 0.0, f"Regression output below 0: {out.min()}"
        assert out.max() <= 1.0, f"Regression output above 1: {out.max()}"


# ── Full model output shape tests ──────────────────────────────────────────────


class TestOutputShapes:

    def test_pred_xy_shape(self):
        model = STTransformer(small_config())
        out = model(random_batch())
        assert out["pred_xy"].shape == (
            BATCH,
            2,
        ), f"pred_xy shape: expected ({BATCH}, 2), got {out['pred_xy'].shape}"

    def test_gesture_logits_shape(self):
        model = STTransformer(small_config())
        out = model(random_batch())
        assert out["gesture_logits"].shape == (BATCH, 2), (
            f"gesture_logits shape: expected ({BATCH}, 2), "
            f"got {out['gesture_logits'].shape}"
        )

    def test_pred_xy_dtype(self):
        model = STTransformer(small_config())
        out = model(random_batch())
        assert out["pred_xy"].dtype == torch.float32

    def test_gesture_logits_dtype(self):
        model = STTransformer(small_config())
        out = model(random_batch())
        assert out["gesture_logits"].dtype == torch.float32

    def test_pred_xy_in_unit_range(self):
        """Sigmoid on regression head guarantees [0, 1] output."""
        model = STTransformer(small_config())
        model.eval()
        with torch.no_grad():
            out = model(random_batch(B=32))
        xy = out["pred_xy"]
        assert xy.min() >= 0.0, f"pred_xy below 0: {xy.min()}"
        assert xy.max() <= 1.0, f"pred_xy above 1: {xy.max()}"

    def test_batch_size_one(self):
        """Model must handle batch size = 1 (used at inference time)."""
        model = STTransformer(small_config())
        model.eval()
        with torch.no_grad():
            out = model(random_batch(B=1))
        assert out["pred_xy"].shape == (1, 2)
        assert out["gesture_logits"].shape == (1, 2)

    def test_variable_batch_size(self):
        """Output shape must scale correctly with batch size."""
        model = STTransformer(small_config())
        model.eval()
        for B in [1, 4, 8, 16]:
            with torch.no_grad():
                out = model(random_batch(B=B))
            assert out["pred_xy"].shape[0] == B
            assert out["gesture_logits"].shape[0] == B


# ── Gradient flow tests ────────────────────────────────────────────────────────


class TestGradientFlow:

    def test_gradients_flow_to_input_proj(self):
        """All parameters must receive gradients after a backward pass."""
        model = STTransformer(small_config())
        out = model(random_batch())
        loss = out["pred_xy"].sum() + out["gesture_logits"].sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.any(
                    torch.isnan(param.grad)
                ), f"NaN gradient for {name}"

    def test_cls_token_gets_gradient(self):
        model = STTransformer(small_config())
        out = model(random_batch())
        loss = out["gesture_logits"].sum()
        loss.backward()
        assert model.cls_token.grad is not None, "CLS token must receive gradient"

    def test_regression_head_gradient(self):
        model = STTransformer(small_config())
        out = model(random_batch())
        out["pred_xy"].sum().backward()
        for name, p in model.regression_head.named_parameters():
            assert p.grad is not None, f"regression_head.{name} has no gradient"

    def test_gesture_head_gradient(self):
        model = STTransformer(small_config())
        out = model(random_batch())
        out["gesture_logits"].sum().backward()
        for name, p in model.gesture_head.named_parameters():
            assert p.grad is not None, f"gesture_head.{name} has no gradient"


# ── Loss function tests ────────────────────────────────────────────────────────


class TestAirSketchLoss:

    def test_loss_output_keys(self):
        loss_fn = AirSketchLoss()
        out = loss_fn(
            pred_xy=torch.rand(BATCH, 2),
            gt_xy=torch.rand(BATCH, 2),
            gesture_logits=torch.randn(BATCH, 2),
            gesture_labels=torch.randint(0, 2, (BATCH,)),
        )
        assert "loss" in out
        assert "regression_loss" in out
        assert "gesture_loss" in out

    def test_loss_is_scalar(self):
        loss_fn = AirSketchLoss()
        out = loss_fn(
            pred_xy=torch.rand(BATCH, 2),
            gt_xy=torch.rand(BATCH, 2),
            gesture_logits=torch.randn(BATCH, 2),
            gesture_labels=torch.randint(0, 2, (BATCH,)),
        )
        assert out["loss"].shape == torch.Size([])

    def test_loss_is_positive(self):
        loss_fn = AirSketchLoss()
        out = loss_fn(
            pred_xy=torch.rand(BATCH, 2),
            gt_xy=torch.rand(BATCH, 2),
            gesture_logits=torch.randn(BATCH, 2),
            gesture_labels=torch.randint(0, 2, (BATCH,)),
        )
        assert out["loss"].item() > 0

    def test_perfect_regression_zero_loss(self):
        """When pred_xy == gt_xy, regression loss should be exactly 0."""
        loss_fn = AirSketchLoss(gesture_loss_weight=0.0)
        xy = torch.rand(BATCH, 2)
        out = loss_fn(
            pred_xy=xy,
            gt_xy=xy,
            gesture_logits=torch.randn(BATCH, 2),
            gesture_labels=torch.zeros(BATCH, dtype=torch.int64),
        )
        assert out["regression_loss"].item() < 1e-6

    def test_loss_lambda_scales_gesture_term(self):
        """Doubling lambda should roughly double the gesture contribution."""
        torch.manual_seed(0)
        logits = torch.randn(BATCH, 2)
        labels = torch.randint(0, 2, (BATCH,))
        xy_p = torch.rand(BATCH, 2)
        xy_g = torch.rand(BATCH, 2)

        loss_lam1 = AirSketchLoss(gesture_loss_weight=0.5)
        loss_lam2 = AirSketchLoss(gesture_loss_weight=1.0)

        out1 = loss_lam1(xy_p, xy_g, logits, labels)
        out2 = loss_lam2(xy_p, xy_g, logits, labels)

        assert (
            out2["loss"].item() > out1["loss"].item()
        ), "Higher lambda should produce higher total loss"

    def test_class_weights_accepted(self):
        """Loss function should accept and use class weights without error."""
        weights = torch.tensor([1.0, 3.0])  # upweight draw class
        loss_fn = AirSketchLoss(class_weights=weights)
        out = loss_fn(
            pred_xy=torch.rand(BATCH, 2),
            gt_xy=torch.rand(BATCH, 2),
            gesture_logits=torch.randn(BATCH, 2),
            gesture_labels=torch.randint(0, 2, (BATCH,)),
        )
        assert torch.isfinite(out["loss"])

    def test_loss_backward(self):
        """Joint loss must support backward pass through the full model."""
        model = STTransformer(small_config())
        loss_fn = AirSketchLoss()
        out = model(random_batch())
        losses = loss_fn(
            pred_xy=out["pred_xy"],
            gt_xy=torch.rand(BATCH, 2),
            gesture_logits=out["gesture_logits"],
            gesture_labels=torch.randint(0, 2, (BATCH,)),
        )
        losses["loss"].backward()
        # Verify no NaN gradients
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.any(torch.isnan(p.grad)), f"NaN gradient: {name}"


# ── Config validation tests ────────────────────────────────────────────────────


class TestConfig:

    def test_invalid_hidden_dim_raises(self):
        """hidden_dim must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            STTransformerConfig(hidden_dim=33, num_heads=4)  # 33 % 4 != 0

    def test_valid_configs_construct(self):
        """Spot-check several valid configs from the sweep."""
        for hidden, layers in [(64, 2), (128, 4), (256, 6)]:
            cfg = STTransformerConfig(hidden_dim=hidden, num_layers=layers)
            model = STTransformer(cfg)
            assert model.count_parameters() > 0

    def test_from_yaml(self):
        config = {
            "model": {
                "hidden_dim": 64,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1,
            },
            "data": {"window_size": 16},
        }
        cfg = STTransformerConfig.from_yaml(config)
        assert cfg.hidden_dim == 64
        assert cfg.num_layers == 2


# ── Parameter count smoke test ─────────────────────────────────────────────────


class TestParameterCount:

    def test_default_config_parameter_range(self):
        """Default config (hidden=128, layers=4) should be 0.5M–5M params."""
        cfg = STTransformerConfig(hidden_dim=128, num_layers=4, num_heads=4)
        model = STTransformer(cfg)
        total = model.count_parameters()
        assert (
            500_000 <= total <= 5_000_000
        ), f"Default model has {total:,} params — expected 500K–5M"

    def test_larger_config_has_more_params(self):
        small = STTransformer(STTransformerConfig(hidden_dim=64, num_layers=2))
        large = STTransformer(STTransformerConfig(hidden_dim=256, num_layers=6))
        assert large.count_parameters() > small.count_parameters()
