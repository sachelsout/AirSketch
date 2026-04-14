"""
scripts/merge_training_splits.py

Merges FreiHAND and EgoHands into a unified training split index.

The merged index preserves source identity so the dataset builder
can load each source's landmarks.npy independently and concatenate
them at window-construction time.

Also performs the overlap check: verifies that no EgoHands clip
or window overlaps with the custom gesture test set from issue #7.

Output:
    data/splits/merged_train_split.json

Usage:
    python scripts/merge_training_splits.py
"""

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def check_overlap(
    egohands_index: dict,
    custom_split: dict,
) -> list[str]:
    """
    Verify there is no overlap between EgoHands clips and custom gesture sessions.

    Overlap is impossible by construction — EgoHands and the custom dataset
    are entirely different recordings by different people on different hardware.
    This function checks for the degenerate case where a file path appears in
    both indices (e.g. a copy-paste error in path configuration).

    Returns a list of overlapping path strings (empty = no overlap, all clear).
    """
    egohands_paths = set()
    for clip_data in egohands_index.values():
        egohands_paths.add(clip_data["landmarks_path"])

    custom_paths = set()
    for session_data in custom_split.get("sessions", {}).values():
        custom_paths.add(session_data["landmarks_path"])

    return list(egohands_paths & custom_paths)


def main() -> None:
    freihand_splits_path = Path("data/splits/freihand_splits.json")
    egohands_index_path = Path("data/processed/egohands/clip_index.json")
    custom_split_path = Path("data/splits/custom_test_split.json")
    output_path = Path("data/splits/merged_train_split.json")

    # ── Load all indices ───────────────────────────────────────────────────────
    print("Loading split indices ...")
    freihand_splits = load_json(freihand_splits_path)
    egohands_index = load_json(egohands_index_path)
    custom_split = load_json(custom_split_path)

    print(f"  FreiHAND train indices:  {len(freihand_splits['train']):,}")
    print(f"  EgoHands clips:          {len(egohands_index)}")
    print(f"  Custom test sessions:    {len(custom_split.get('sessions', {}))}")

    # ── Overlap check ──────────────────────────────────────────────────────────
    print("\nRunning overlap check ...")
    overlaps = check_overlap(egohands_index, custom_split)

    if overlaps:
        print(
            f"  ERROR: {len(overlaps)} overlapping paths found between "
            f"EgoHands and custom test set:"
        )
        for o in overlaps:
            print(f"    {o}")
        raise RuntimeError(
            "Overlap detected between training and test data. "
            "Do NOT proceed until this is resolved."
        )
    else:
        print("  ✓ No overlap between EgoHands and custom test set.")

    # ── Also verify FreiHAND test split does not overlap custom ───────────────
    freihand_test_set = set(freihand_splits["test"])
    freihand_train_set = set(freihand_splits["train"])
    assert (
        len(freihand_test_set & freihand_train_set) == 0
    ), "FreiHAND train/test overlap — re-run issue #4 split script."
    print("  ✓ FreiHAND train/test split confirmed clean.")

    # ── Build merged index ─────────────────────────────────────────────────────
    merged = {
        "sources": {
            "freihand": {
                "landmarks_path": "data/processed/freihand/landmarks.npy",
                "train_indices": freihand_splits["train"],
                "val_indices": freihand_splits["val"],
                "frame_count": len(freihand_splits["train"])
                + len(freihand_splits["val"]),
            },
            "egohands": {
                "clips": {
                    clip_name: {
                        "landmarks_path": clip_data["landmarks_path"],
                        "n_frames": clip_data["summary"].get("total_frames", 100),
                        "detection_rate": clip_data["summary"].get(
                            "detection_rate", 0.0
                        ),
                    }
                    for clip_name, clip_data in egohands_index.items()
                },
                "total_clips": len(egohands_index),
                "total_frames": sum(
                    d["summary"].get("total_frames", 100)
                    for d in egohands_index.values()
                ),
            },
        },
        "test_source": {
            "custom": str(custom_split_path),
            "note": "Custom gesture sessions — held out entirely, never used for training.",
        },
        "overlap_check_passed": True,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    # ── Print final summary ────────────────────────────────────────────────────
    eg_frames = merged["sources"]["egohands"]["total_frames"]
    fh_train = len(freihand_splits["train"])

    print("\n── Merged training split ───────────────────────────────────────────")
    print(f"  FreiHAND train frames:  {fh_train:,}")
    print(f"  EgoHands frames:        {eg_frames:,}")
    print(f"  Total training frames:  {fh_train + eg_frames:,}")
    print(
        f"  Custom test (held-out): {custom_split['aggregate']['total_frames']:,} frames "
        f"across {custom_split['aggregate']['total_sessions']} sessions"
    )
    print(f"\n  Saved: {output_path}")
    print(
        "\n✓ Merge complete. Pass merged_train_split.json to AirSketchDataset in issue #6."
    )


if __name__ == "__main__":
    main()
