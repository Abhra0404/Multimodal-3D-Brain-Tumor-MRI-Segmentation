from pathlib import Path
import json

import nibabel as nib
import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "v2"
)


# --------------------------------------------------
# Load patient split
# --------------------------------------------------

with open(
    METADATA_DIR / "patient_split.json"
) as f:
    split = json.load(f)


train_ids = split["train_ids"]
val_ids = split["val_ids"]


# --------------------------------------------------
# Build index
# --------------------------------------------------

def build_index(patient_ids):

    positive = []
    negative = []

    for i, patient_id in enumerate(
        patient_ids,
        start=1
    ):

        seg_path = (
            DATASET_DIR
            / patient_id
            / f"{patient_id}-seg.nii.gz"
        )

        seg = nib.load(seg_path).get_fdata()

        tumor_slices = np.any(
            seg > 0,
            axis=(0, 1)
        )

        positive_slices = np.where(
            tumor_slices
        )[0]

        negative_slices = np.where(
            ~tumor_slices
        )[0]

        for slice_idx in positive_slices:
            positive.append(
                [
                    patient_id,
                    int(slice_idx)
                ]
            )

        for slice_idx in negative_slices:
            negative.append(
                [
                    patient_id,
                    int(slice_idx)
                ]
            )

        if i % 100 == 0 or i == len(patient_ids):
            print(
                f"Processed {i}/{len(patient_ids)}"
            )

    return positive, negative


print("Building training index...")

train_positive, train_negative = build_index(
    train_ids
)

print()
print("Building validation index...")

val_positive, val_negative = build_index(
    val_ids
)


# --------------------------------------------------
# Save
# --------------------------------------------------

index = {
    "train": {
        "positive": train_positive,
        "negative": train_negative,
    },
    "val": {
        "positive": val_positive,
        "negative": val_negative,
    },
}


output_path = (
    METADATA_DIR
    / "slice_index.json"
)

with open(
    output_path,
    "w"
) as f:
    json.dump(
        index,
        f
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("=" * 50)
print("Slice index complete")
print("=" * 50)

print(
    "Training positive:",
    len(train_positive)
)

print(
    "Training negative:",
    len(train_negative)
)

print(
    "Validation positive:",
    len(val_positive)
)

print(
    "Validation negative:",
    len(val_negative)
)

print()
print("Saved:", output_path)