from pathlib import Path
import json

import numpy as np


# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
)

METADATA_DIR = (
    PROCESSED_DIR
    / "metadata"
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# Build index
# ==================================================

def build_index(
    split
):

    data_dir = (
        PROCESSED_DIR
        / split
    )

    files = sorted(
        data_dir.glob("*.npz")
    )

    tumor_indices = []
    background_indices = []

    print()
    print(
        f"Building {split} patch index..."
    )

    for index, path in enumerate(files):

        data = np.load(
            path
        )

        mask = data["mask"]

        tumor_voxels = int(
            mask.sum()
        )

        data.close()

        if tumor_voxels > 0:

            tumor_indices.append(
                index
            )

        else:

            background_indices.append(
                index
            )

    index_data = {
        "split": split,
        "total_patches": len(files),
        "tumor_patches": len(
            tumor_indices
        ),
        "background_patches": len(
            background_indices
        ),
        "files": [
            str(path.name)
            for path in files
        ],
        "tumor_indices": tumor_indices,
        "background_indices": background_indices
    }

    output_path = (
        METADATA_DIR
        / f"{split}_patch_index.json"
    )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            index_data,
            f,
            indent=2
        )

    print(
        "Total:",
        len(files)
    )

    print(
        "Tumor:",
        len(tumor_indices)
    )

    print(
        "Background:",
        len(background_indices)
    )

    print(
        "Saved:",
        output_path
    )


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    build_index("train")
    build_index("val")
    