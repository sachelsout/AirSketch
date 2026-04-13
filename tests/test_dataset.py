"""
tests/test_dataset.py

Unit tests for AirSketchDataset.
Uses a synthetic in-memory landmark array — no real .npy files required.
"""

import tempfile

import numpy as np
import torch

from src.dataset import AirSketchDataset, WINDOW_SIZE, FEATURE_DIM, INDEX_FINGERTIP


# ── Fixtures ───────────────────────────────────────────────────────────────────


def make_landmarks(n: int, nan_frames: list[int] | None = None) -> np.ndarray:
    """Create a synthetic (n, 21, 2) float32 landmark array."""
    rng = np.random.default_rng(42)
    lm = rng.random((n, 21, 2)).astype(np.float32)
    if nan_frames:
        for i in nan_frames:
            lm[i] = np.nan
    return lm


def make_dataset(
    n_frames: int = 200,
    nan_frames: list | None = None,
    augment: bool = False,
    stride: int = 1,
) -> AirSketchDataset:
    """Write a temp landmarks.npy and return an AirSketchDataset over all frames."""
    lm = make_landmarks(n_frames, nan_frames)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, lm)
        return AirSketchDataset(
            landmarks_path=f.name,
            split_indices=list(range(n_frames)),
            window_size=WINDOW_SIZE,
            stride=stride,
            augment=augment,
        )


# ── Shape and type tests ───────────────────────────────────────────────────────


class TestSampleShape:

    def test_sequence_shape(self):
        ds = make_dataset()
        sample = ds[0]
        assert sample["sequence"].shape == (
            WINDOW_SIZE,
            FEATURE_DIM,
        ), f"Expected sequence shape (16, 42), got {sample['sequence'].shape}"

    def test_target_shape(self):
        ds = make_dataset()
        sample = ds[0]
        assert sample["target"].shape == (
            2,
        ), f"Expected target shape (2,), got {sample['target'].shape}"

    def test_gesture_is_scalar(self):
        ds = make_dataset()
        sample = ds[0]
        assert sample["gesture"].shape == torch.Size(
            []
        ), f"Expected scalar gesture, got shape {sample['gesture'].shape}"

    def test_sequence_dtype(self):
        ds = make_dataset()
        sample = ds[0]
        assert sample["sequence"].dtype == torch.float32

    def test_target_dtype(self):
        ds = make_dataset()
        sample = ds[0]
        assert sample["target"].dtype == torch.float32

    def test_gesture_dtype(self):
        ds = make_dataset()
        sample = ds[0]
        assert sample["gesture"].dtype == torch.int64

    def test_no_nans_in_sample(self):
        ds = make_dataset()
        sample = ds[0]
        assert not torch.any(torch.isnan(sample["sequence"])), "NaN in sequence"
        assert not torch.any(torch.isnan(sample["target"])), "NaN in target"


# ── Value range tests ──────────────────────────────────────────────────────────


class TestValueRanges:

    def test_sequence_in_unit_range(self):
        ds = make_dataset()
        for i in range(min(50, len(ds))):
            seq = ds[i]["sequence"]
            assert seq.min() >= 0.0, f"sequence[{i}] below 0: {seq.min()}"
            assert seq.max() <= 1.0, f"sequence[{i}] above 1: {seq.max()}"

    def test_target_in_unit_range(self):
        ds = make_dataset()
        for i in range(min(50, len(ds))):
            tgt = ds[i]["target"]
            assert tgt.min() >= 0.0, f"target[{i}] below 0: {tgt.min()}"
            assert tgt.max() <= 1.0, f"target[{i}] above 1: {tgt.max()}"

    def test_gesture_binary(self):
        ds = make_dataset()
        for i in range(min(50, len(ds))):
            g = ds[i]["gesture"].item()
            assert g in (0, 1), f"gesture[{i}] not binary: {g}"


# ── Window count tests ─────────────────────────────────────────────────────────


class TestWindowCounts:

    def test_expected_window_count_no_gaps(self):
        n = 100
        ds = make_dataset(n_frames=n, stride=1)
        # With no NaNs: n - (T+1) + 1 = n - T = 100 - 16 = 84 windows
        expected = n - WINDOW_SIZE
        assert len(ds) == expected, f"Expected {expected} windows, got {len(ds)}"

    def test_stride_reduces_window_count(self):
        ds_s1 = make_dataset(n_frames=100, stride=1)
        ds_s4 = make_dataset(n_frames=100, stride=4)
        assert len(ds_s4) < len(ds_s1), "Larger stride should produce fewer windows"

    def test_gap_reduces_window_count(self):
        ds_no_gap = make_dataset(n_frames=100, nan_frames=None)
        ds_gap = make_dataset(n_frames=100, nan_frames=list(range(40, 60)))
        assert len(ds_gap) < len(ds_no_gap), "Gaps should reduce window count"

    def test_all_nan_produces_empty_dataset(self):
        ds = make_dataset(n_frames=50, nan_frames=list(range(50)))
        assert len(ds) == 0, "All-NaN input should produce zero windows"

    def test_too_short_produces_empty_dataset(self):
        ds = make_dataset(n_frames=10)  # shorter than WINDOW_SIZE + 1
        assert len(ds) == 0, "Sequence shorter than T+1 should produce zero windows"


# ── Augmentation tests ─────────────────────────────────────────────────────────


class TestAugmentation:

    def test_augmented_sequence_in_unit_range(self):
        ds = make_dataset(augment=True)
        for i in range(min(100, len(ds))):
            seq = ds[i]["sequence"]
            assert (
                seq.min() >= 0.0 and seq.max() <= 1.0
            ), f"Augmented sequence[{i}] out of [0,1]: min={seq.min()}, max={seq.max()}"

    def test_augmented_target_in_unit_range(self):
        ds = make_dataset(augment=True)
        for i in range(min(100, len(ds))):
            tgt = ds[i]["target"]
            assert tgt.min() >= 0.0 and tgt.max() <= 1.0

    def test_flip_changes_x_consistently(self):
        """
        When a horizontal flip is applied, the x coordinate of every landmark
        in the window should equal 1 - original_x. Verify by disabling noise
        (sigma=0) and forcing flip_prob=1.0.
        """
        lm = make_landmarks(50)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, lm)
            ds_orig = AirSketchDataset(f.name, list(range(50)), augment=False)
            ds_flip = AirSketchDataset(
                f.name, list(range(50)), augment=True, flip_prob=1.0, noise_sigma=0.0
            )

        orig = ds_orig[0]["sequence"].numpy().reshape(WINDOW_SIZE, 21, 2)
        flip = ds_flip[0]["sequence"].numpy().reshape(WINDOW_SIZE, 21, 2)

        np.testing.assert_allclose(
            flip[:, :, 0],
            1.0 - orig[:, :, 0],
            atol=1e-6,
            err_msg="Flipped x should equal 1 - original x",
        )
        np.testing.assert_allclose(
            flip[:, :, 1],
            orig[:, :, 1],
            atol=1e-6,
            err_msg="y coordinates should be unchanged by horizontal flip",
        )

    def test_no_augmentation_when_augment_false(self):
        """Two calls to __getitem__ with the same index should return identical data."""
        ds = make_dataset(augment=False)
        s1 = ds[0]["sequence"]
        s2 = ds[0]["sequence"]
        torch.testing.assert_close(s1, s2)


# ── Gesture label tests ────────────────────────────────────────────────────────


class TestGestureLabels:

    def test_static_hand_is_idle(self):
        """A completely static hand (all frames identical) should be labelled idle."""
        lm = np.zeros((50, 21, 2), dtype=np.float32)
        lm[:, :, :] = 0.5  # fixed position — zero velocity
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, lm)
            ds = AirSketchDataset(f.name, list(range(50)), augment=False)
        for i in range(len(ds)):
            assert (
                ds[i]["gesture"].item() == 0
            ), f"Static hand should be idle at window {i}"

    def test_fast_moving_hand_is_draw(self):
        """A rapidly moving fingertip should be labelled draw."""
        lm = np.zeros((50, 21, 2), dtype=np.float32)
        # Make fingertip move ~0.02 per frame — well above threshold
        lm[:, INDEX_FINGERTIP, 0] = np.linspace(0.1, 0.9, 50)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f.name, lm)
            ds = AirSketchDataset(f.name, list(range(50)), augment=False)
        draw_count = sum(ds[i]["gesture"].item() for i in range(len(ds)))
        assert (
            draw_count > 0
        ), "Fast-moving fingertip should produce at least one draw label"


# ── DataLoader integration test ────────────────────────────────────────────────


class TestDataLoader:

    def test_dataloader_batch_shapes(self):
        from torch.utils.data import DataLoader

        ds = make_dataset(n_frames=200)
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        batch = next(iter(loader))

        assert batch["sequence"].shape == (8, WINDOW_SIZE, FEATURE_DIM)
        assert batch["target"].shape == (8, 2)
        assert batch["gesture"].shape == (8,)

    def test_dataloader_no_nans(self):
        from torch.utils.data import DataLoader

        ds = make_dataset(n_frames=200)
        loader = DataLoader(ds, batch_size=16)
        for batch in loader:
            assert not torch.any(
                torch.isnan(batch["sequence"])
            ), "NaN in batch sequence"
            assert not torch.any(torch.isnan(batch["target"])), "NaN in batch target"
