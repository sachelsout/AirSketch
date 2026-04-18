"""
tests/test_train.py

Unit tests for the training loop.
Uses tiny synthetic data and small model configs for speed.
All tests run on CPU.
"""

import numpy as np
import torch
import yaml

from src.model import STTransformer, STTransformerConfig, AirSketchLoss
from src.train import (
    load_config,
    save_checkpoint,
    load_checkpoint,
    run_epoch,
    _targets_met,
    set_seed,
)

BATCH = 4


def small_model() -> STTransformer:
    cfg = STTransformerConfig(hidden_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    return STTransformer(cfg)


def fake_loader(n_batches: int = 3) -> list[dict]:
    return [
        {
            "sequence": torch.rand(BATCH, 16, 42),
            "target": torch.rand(BATCH, 2),
            "gesture": torch.randint(0, 2, (BATCH,)),
        }
        for _ in range(n_batches)
    ]


def make_optimizer(model) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


def make_scheduler(opt) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)


class TestSeed:

    def test_reproducibility(self):
        set_seed(42)
        a = torch.rand(5)
        set_seed(42)
        b = torch.rand(5)
        torch.testing.assert_close(a, b)


class TestLoadConfig:

    def _write_config(self, tmp_path):
        cfg = {
            "model": {"hidden_dim": 128, "num_layers": 4},
            "training": {"learning_rate": 0.001, "epochs": 50},
        }
        p = tmp_path / "cfg.yaml"
        with open(p, "w") as f:
            yaml.dump(cfg, f)
        return p

    def test_loads_base(self, tmp_path):
        cfg = load_config(str(self._write_config(tmp_path)))
        assert cfg["model"]["hidden_dim"] == 128

    def test_integer_override(self, tmp_path):
        cfg = load_config(
            str(self._write_config(tmp_path)), overrides=["model.hidden_dim=256"]
        )
        assert cfg["model"]["hidden_dim"] == 256
        assert isinstance(cfg["model"]["hidden_dim"], int)

    def test_float_override(self, tmp_path):
        cfg = load_config(
            str(self._write_config(tmp_path)),
            overrides=["training.learning_rate=0.0005"],
        )
        assert abs(cfg["training"]["learning_rate"] - 0.0005) < 1e-10

    def test_multiple_overrides(self, tmp_path):
        cfg = load_config(
            str(self._write_config(tmp_path)),
            overrides=["model.hidden_dim=64", "model.num_layers=2"],
        )
        assert cfg["model"]["hidden_dim"] == 64
        assert cfg["model"]["num_layers"] == 2


class TestCheckpoint:

    def test_save_and_load(self, tmp_path):
        model = small_model()
        optimizer = make_optimizer(model)
        scheduler = make_scheduler(optimizer)
        path = tmp_path / "ckpt.pt"

        save_checkpoint(
            path, model, optimizer, scheduler, epoch=5, best_mpjpe=6.42, config={}
        )

        model2 = small_model()
        optimizer2 = make_optimizer(model2)
        epoch, mpjpe = load_checkpoint(path, model2, optimizer2)

        assert epoch == 5
        assert abs(mpjpe - 6.42) < 1e-6

        for key in model.state_dict():
            torch.testing.assert_close(
                model.state_dict()[key], model2.state_dict()[key]
            )

    def test_load_without_optimizer(self, tmp_path):
        model = small_model()
        optimizer = make_optimizer(model)
        scheduler = make_scheduler(optimizer)
        path = tmp_path / "ckpt.pt"

        save_checkpoint(
            path, model, optimizer, scheduler, epoch=3, best_mpjpe=7.1, config={}
        )

        model2 = small_model()
        epoch, mpjpe = load_checkpoint(path, model2)
        assert epoch == 3


class TestRunEpoch:

    def test_returns_expected_keys(self):
        model = small_model()
        optimizer = make_optimizer(model)
        loss_fn = AirSketchLoss()

        metrics = run_epoch(
            model=model,
            loader=fake_loader(),
            loss_fn=loss_fn,
            optimizer=optimizer,
            device="cpu",
            logger=None,
            epoch=1,
            is_train=True,
        )

        for key in [
            "loss",
            "regression_loss",
            "gesture_loss",
            "mpjpe",
            "jitter_index",
            "gesture_acc",
        ]:
            assert key in metrics, f"Missing key: {key}"

    def test_val_does_not_modify_params(self):
        model = small_model()
        loss_fn = AirSketchLoss()

        params_before = {k: v.clone() for k, v in model.named_parameters()}

        run_epoch(
            model=model,
            loader=fake_loader(),
            loss_fn=loss_fn,
            optimizer=None,
            device="cpu",
            logger=None,
            epoch=1,
            is_train=False,
        )

        for name, param in model.named_parameters():
            torch.testing.assert_close(
                param,
                params_before[name],
                msg=f"Parameter {name} changed during eval epoch",
            )

    def test_train_modifies_params(self):
        model = small_model()
        optimizer = make_optimizer(model)
        loss_fn = AirSketchLoss()

        params_before = {k: v.clone() for k, v in model.named_parameters()}

        run_epoch(
            model=model,
            loader=fake_loader(),
            loss_fn=loss_fn,
            optimizer=optimizer,
            device="cpu",
            logger=None,
            epoch=1,
            is_train=True,
        )

        changed = any(
            not torch.equal(param, params_before[name])
            for name, param in model.named_parameters()
            if param.requires_grad
        )
        assert changed, "Training epoch must update at least one parameter"

    def test_metrics_are_finite(self):
        model = small_model()
        optimizer = make_optimizer(model)
        loss_fn = AirSketchLoss()

        metrics = run_epoch(
            model=model,
            loader=fake_loader(n_batches=5),
            loss_fn=loss_fn,
            optimizer=optimizer,
            device="cpu",
            logger=None,
            epoch=1,
            is_train=True,
        )

        for key, val in metrics.items():
            assert np.isfinite(val), f"Non-finite metric: {key} = {val}"

    def test_gesture_acc_in_unit_range(self):
        model = small_model()
        loss_fn = AirSketchLoss()

        metrics = run_epoch(
            model=model,
            loader=fake_loader(),
            loss_fn=loss_fn,
            optimizer=None,
            device="cpu",
            logger=None,
            epoch=1,
            is_train=False,
        )

        assert 0.0 <= metrics["gesture_acc"] <= 1.0

    def test_multiple_epochs_consistent_keys(self):
        model = small_model()
        optimizer = make_optimizer(model)
        scheduler = make_scheduler(optimizer)
        loss_fn = AirSketchLoss()

        keys_e1 = set(
            run_epoch(
                model, fake_loader(), loss_fn, optimizer, "cpu", None, 1, True
            ).keys()
        )
        scheduler.step()
        keys_e2 = set(
            run_epoch(
                model, fake_loader(), loss_fn, optimizer, "cpu", None, 2, True
            ).keys()
        )

        assert keys_e1 == keys_e2


class TestTargetsMet:

    def test_all_targets_met(self):
        assert _targets_met({"mpjpe": 7.5, "jitter_index": 1.8, "gesture_acc": 0.93})

    def test_mpjpe_not_met(self):
        assert not _targets_met(
            {"mpjpe": 8.5, "jitter_index": 1.8, "gesture_acc": 0.95}
        )

    def test_jitter_not_met(self):
        assert not _targets_met(
            {"mpjpe": 6.0, "jitter_index": 2.5, "gesture_acc": 0.95}
        )

    def test_gesture_not_met(self):
        assert not _targets_met(
            {"mpjpe": 6.0, "jitter_index": 1.5, "gesture_acc": 0.89}
        )

    def test_boundary_values_fail(self):
        # Strict inequalities -- boundary values must not pass
        assert not _targets_met(
            {"mpjpe": 8.0, "jitter_index": 2.0, "gesture_acc": 0.92}
        )
