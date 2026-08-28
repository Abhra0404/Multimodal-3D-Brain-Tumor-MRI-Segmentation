from pathlib import Path
import math
import time

import numpy as np


# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
)

SHARD_DIR = (
    PROCESSED_DIR
    / "shards"
)


# ==================================================
# Configuration
# ==================================================

PATCHES_PER_SHARD = 500

SPLITS = [
    "train",
    "val"
]


# ==================================================
# Build shards
# ==================================================

def build_shards(split):

    source_dir = (
        PROCESSED_DIR
        / split
    )

    output_dir = (
        SHARD_DIR
        / split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        source_dir.glob("*.npz")
    )

    if not files:

        raise RuntimeError(
            f"No patches found in {source_dir}"
        )

    num_shards = math.ceil(
        len(files)
        / PATCHES_PER_SHARD
    )

    print()
    print("=" * 60)
    print(
        f"BUILDING {split.upper()} SHARDS"
    )
    print("=" * 60)

    print(
        "Patches:",
        len(files)
    )

    print(
        "Patches/shard:",
        PATCHES_PER_SHARD
    )

    print(
        "Expected shards:",
        num_shards
    )

    start_time = time.time()

    for shard_index in range(
        num_shards
    ):

        start = (
            shard_index
            * PATCHES_PER_SHARD
        )

        end = min(
            start + PATCHES_PER_SHARD,
            len(files)
        )

        shard_files = files[
            start:end
        ]

        images = []
        masks = []

        for path in shard_files:

            data = np.load(
                path
            )

            images.append(
                data["image"]
            )

            masks.append(
                data["mask"]
            )

            data.close()

        images = np.stack(
            images,
            axis=0
        ).astype(
            np.float32
        )

        masks = np.stack(
            masks,
            axis=0
        ).astype(
            np.float32
        )

        shard_path = (
            output_dir
            / f"shard_{shard_index:03d}.npz"
        )

        np.savez_compressed(
            shard_path,
            images=images,
            masks=masks
        )

        print(
            f"[{shard_index + 1}/{num_shards}] "
            f"{len(shard_files)} patches "
            f"→ {shard_path.name}"
        )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"{split.upper()} SHARDS COMPLETE"
    )

    print(
        "Total patches:",
        len(files)
    )

    print(
        "Total shards:",
        num_shards
    )

    print(
        f"Time: {elapsed / 60:.2f} min"
    )

    print(
        "Output:",
        output_dir
    )


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    for split in SPLITS:

        build_shards(
            split
        )