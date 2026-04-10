"""
Validate and preprocess the FreiHAND dataset for AirSketch.

Steps:
1. Verify all expected images exist and are readable.
2. Load training_K.json and training_xyz.json.
3. Project 3D keypoints to 2D pixel coordinates using camera intrinsics.
4. Validate projected landmarks.
5. Save processed landmarks as one .npy array.
6. Create leak-safe 70/15/15 scene-level train/val/test splits.
7. Save split indices as JSON.

Usage:
    python scripts/freihand_download.py \
        --data-dir data/raw/freihand \
        --output-dir data/processed/freihand
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


# Constants
NUM_IMAGES = 130_240
NUM_UNIQUE_SCENES = 32_560
AUGMENTATIONS = 4
NUM_LANDMARKS = 21
IMAGE_SIZE = 224
INDEX_FINGERTIP = 8
SPLIT_RATIOS = (0.70, 0.15, 0.15)
RANDOM_SEED = 42


def project_3d_to_2d(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3D camera-space landmarks to normalized 2D image coordinates."""
    uvw = (K @ xyz.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    uv = uv / IMAGE_SIZE
    return uv.astype(np.float32)


def validate_landmarks(uv: np.ndarray, image_idx: int) -> list[str]:
    """Run sanity checks for one image's projected landmarks."""
    errors: list[str] = []

    if uv.shape != (NUM_LANDMARKS, 2):
        errors.append(f"[{image_idx}] Wrong shape: {uv.shape}, expected (21, 2)")

    if np.any(np.isnan(uv)):
        errors.append(f"[{image_idx}] NaN values in landmarks")

    if np.any(np.isinf(uv)):
        errors.append(f"[{image_idx}] Inf values in landmarks")

    # Allow mild overshoot for partially occluded landmarks.
    if np.any(uv < -0.15) or np.any(uv > 1.15):
        errors.append(
            f"[{image_idx}] Landmarks out of expected range: "
            f"min={uv.min():.3f}, max={uv.max():.3f}"
        )

    return errors


def verify_images(rgb_dir: Path, num_images: int) -> tuple[int, list[int]]:
    """Check all expected images exist by filename only (no decode)."""
    missing: list[int] = []
    print(f"Verifying {num_images:,} images in {rgb_dir} ...")

    for i in tqdm(range(num_images), desc="Checking images", unit="img"):
        img_path = rgb_dir / f"{i:08d}.jpg"  # FreiHAND uses 8-digit filenames
        if not img_path.exists():
            missing.append(i)

    return num_images - len(missing), missing


def make_split_indices(
    num_scenes: int = NUM_UNIQUE_SCENES,
    augmentations: int = AUGMENTATIONS,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
) -> dict[str, list[int]]:
    """Create leak-safe scene-level 70/15/15 train/val/test image index splits."""
    rng = np.random.default_rng(seed)
    scenes = np.arange(num_scenes)
    rng.shuffle(scenes)

    n_train = int(num_scenes * ratios[0])
    n_val = int(num_scenes * ratios[1])

    train_scenes = scenes[:n_train]
    val_scenes = scenes[n_train : n_train + n_val]
    test_scenes = scenes[n_train + n_val :]

    def scenes_to_image_indices(scene_ids: np.ndarray) -> list[int]:
        indices: list[int] = []
        for scene_id in scene_ids:
            for aug in range(augmentations):
                indices.append(int(scene_id + aug * num_scenes))
        return sorted(indices)

    splits = {
        "train": scenes_to_image_indices(train_scenes),
        "val": scenes_to_image_indices(val_scenes),
        "test": scenes_to_image_indices(test_scenes),
    }

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert len(train_set & val_set) == 0, "Overlap between train and val"
    assert len(train_set & test_set) == 0, "Overlap between train and test"
    assert len(val_set & test_set) == 0, "Overlap between val and test"
    assert len(train_set) + len(val_set) + len(test_set) == num_scenes * augmentations

    return splits


def main(data_dir: str, output_dir: str) -> None:
    data_dir_path = Path(data_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # JSONs live at the freihand root; images are under training/rgb/
    rgb_dir = data_dir_path / "training" / "rgb"
    K_path = data_dir_path / "training_K.json"
    xyz_path = data_dir_path / "training_xyz.json"

    print("\n-- Step 1: Image verification --")
    num_valid, missing = verify_images(rgb_dir, NUM_IMAGES)
    if missing:
        print(f"WARNING: {len(missing)} images missing or corrupt")
        print(f"First 10 missing indices: {missing[:10]}")
    else:
        print(f"[OK] All {num_valid:,} images verified")

    print("\n-- Step 2: Loading annotations --")
    print(f"Loading camera intrinsics from {K_path} ...")
    with K_path.open("r", encoding="utf-8") as f:
        K_list = json.load(f)

    print(f"Loading 3D keypoints from {xyz_path} ...")
    with xyz_path.open("r", encoding="utf-8") as f:
        xyz_list = json.load(f)

    # JSONs have one entry per unique scene (32,560), not per image (130,240).
    # The 4 background augmentations of each scene share the same K and xyz.
    assert (
        len(K_list) == NUM_UNIQUE_SCENES
    ), f"Expected {NUM_UNIQUE_SCENES} K matrices, got {len(K_list)}"
    assert (
        len(xyz_list) == NUM_UNIQUE_SCENES
    ), f"Expected {NUM_UNIQUE_SCENES} xyz arrays, got {len(xyz_list)}"
    print(f"[OK] Annotations loaded: {len(K_list):,} entries (one per unique scene)")

    print("\n-- Step 3: Projecting 3D to 2D and validating --")
    all_landmarks = np.zeros((NUM_IMAGES, NUM_LANDMARKS, 2), dtype=np.float32)
    all_errors: list[str] = []

    for i in tqdm(range(NUM_IMAGES), desc="Projecting", unit="img"):
        # All 4 augmentations of a scene share the same annotation.
        scene_id = i % NUM_UNIQUE_SCENES
        K = np.array(K_list[scene_id], dtype=np.float64)
        xyz = np.array(xyz_list[scene_id], dtype=np.float64)

        uv = project_3d_to_2d(xyz, K)
        all_landmarks[i] = uv

        errors = validate_landmarks(uv, i)
        all_errors.extend(errors)

    if all_errors:
        print(f"WARNING: {len(all_errors)} validation errors found")
        for err in all_errors[:20]:
            print(f"  {err}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")
    else:
        print(f"[OK] All {NUM_IMAGES:,} landmark projections passed validation")

    print("\n-- Step 4: Saving processed landmarks --")
    landmarks_path = output_dir_path / "landmarks_2d.npy"
    np.save(landmarks_path, all_landmarks)
    size_mb = landmarks_path.stat().st_size / 1e6
    print(f"[OK] Saved: {landmarks_path} ({size_mb:.1f} MB)")
    print(f"Shape: {all_landmarks.shape} dtype: {all_landmarks.dtype}")
    print(
        "Index fingertip (landmark 8) x range: "
        f"[{all_landmarks[:, INDEX_FINGERTIP, 0].min():.3f}, "
        f"{all_landmarks[:, INDEX_FINGERTIP, 0].max():.3f}]"
    )

    print("\n-- Step 5: Creating train/val/test split --")
    splits = make_split_indices()

    for split_name, indices in splits.items():
        pct = len(indices) / NUM_IMAGES * 100
        print(f"{split_name:5s}: {len(indices):7,} images ({pct:.1f}%)")

    splits_path = Path("data/splits/freihand_splits.json")
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f)
    print(f"[OK] Splits saved: {splits_path}")

    print("\n-- Summary --")
    print(f"Total images:    {NUM_IMAGES:,}")
    print(f"Valid images:    {num_valid:,}")
    print(f"Missing images:  {len(missing)}")
    print(f"Landmark errors: {len(all_errors)}")
    print(f"Train split:     {len(splits['train']):,}")
    print(f"Val split:       {len(splits['val']):,}")
    print(f"Test split:      {len(splits['test']):,}")

    if missing or all_errors:
        print("\nWARNING: Issues found. Review errors before proceeding to issue #5.")
        sys.exit(1)

    print("\n[OK] Dataset ready. Proceed to issue #5.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and split FreiHAND dataset")
    parser.add_argument(
        "--data-dir",
        default="data/raw/freihand",
        help="Path to FreiHAND root directory (contains training_K.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/freihand",
        help="Where to save processed landmarks and split files",
    )
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
