from pathlib import Path
import json

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


# ==================================================
# Paths
# ==================================================

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v2"
)


# ==================================================
# Configuration
# ==================================================

TARGET_SIZE = 128
SHARD_SIZE = 1000

MODALITIES = [
    "t1n",
    "t1c",
    "t2w",
    "t2f",
]


# ==================================================
# Load sample metadata
# ==================================================

with open(
    METADATA_DIR / "sampled_slices.json"
) as f:
    sampled = json.load(f)


# ==================================================
# Process one patient
# ==================================================

def load_patient(
    patient_id,
    selected_slices
):
    patient_dir = DATASET_DIR / patient_id

    # ------------------------------
    # Load MRI
    # ------------------------------

    volumes = []

    for modality in MODALITIES:

        path = (
            patient_dir
            / f"{patient_id}-{modality}.nii.gz"
        )

        volume = nib.load(path).get_fdata(
            dtype=np.float32
        )

        volumes.append(volume)

    mri = np.stack(volumes, axis=0)

    # ------------------------------
    # Normalize
    # ------------------------------

    for channel in range(4):

        volume = mri[channel]

        mask = volume > 0

        mean = volume[mask].mean()
        std = volume[mask].std()

        if std > 0:
            volume[mask] = (
                (volume[mask] - mean) / std
            )

        mri[channel] = volume

    # ------------------------------
    # Load segmentation
    # ------------------------------

    seg_path = (
        patient_dir
        / f"{patient_id}-seg.nii.gz"
    )

    seg = nib.load(
        seg_path
    ).get_fdata()

    seg = (seg > 0).astype(np.float32)

    # ------------------------------
    # Select slices first
    # ------------------------------

    selected_slices = np.asarray(
        selected_slices,
        dtype=np.int64
    )

    images = np.transpose(
        mri[:, :, :, selected_slices],
        (3, 0, 1, 2)
    )

    masks = np.transpose(
        seg[:, :, selected_slices],
        (2, 0, 1)
    )

    # ------------------------------
    # Resize entire batch
    # ------------------------------

    scale = TARGET_SIZE / 240

    images_resized = zoom(
        images,
        (
            1,
            1,
            scale,
            scale
        ),
        order=1
    )

    masks_resized = zoom(
        masks,
        (
            1,
            scale,
            scale
        ),
        order=0
    )

    return (
        images_resized.astype(np.float32),
        masks_resized.astype(np.float32)
    )


# ==================================================
# Group samples by patient
# ==================================================

def group_by_patient(
    samples
):

    grouped = {}

    for patient_id, slice_idx in samples:

        grouped.setdefault(
            patient_id,
            []
        ).append(
            slice_idx
        )

    return grouped


# ==================================================
# Preprocess split
# ==================================================

def preprocess_split(
    samples,
    split_name
):

    output_dir = (
        OUTPUT_DIR
        / split_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    grouped = group_by_patient(
        samples
    )

    images_buffer = []
    masks_buffer = []

    shard_id = 0
    total_samples = 0

    for i, (
        patient_id,
        slice_indices
    ) in enumerate(
        grouped.items(),
        start=1
    ):

        print(
            f"[{i}/{len(grouped)}] "
            f"{patient_id} "
            f"({len(slice_indices)} slices)"
        )

        images, masks = load_patient(
            patient_id,
            slice_indices
        )

        images_buffer.append(
            images
        )

        masks_buffer.append(
            masks
        )

        total_samples += len(images)

        current_count = sum(
            len(x)
            for x in images_buffer
        )

        # --------------------------
        # Save shard
        # --------------------------

        if current_count >= SHARD_SIZE:

            shard_images = np.concatenate(
                images_buffer,
                axis=0
            )

            shard_masks = np.concatenate(
                masks_buffer,
                axis=0
            )

            output_path = (
                output_dir
                / f"shard_{shard_id:03d}.npz"
            )

            np.savez(
                output_path,
                images=shard_images,
                masks=shard_masks
            )

            print(
                f"  Saved shard "
                f"{shard_id}: "
                f"{len(shard_images)} samples"
            )

            shard_id += 1

            images_buffer = []
            masks_buffer = []

    # ------------------------------
    # Save remainder
    # ------------------------------

    if images_buffer:

        shard_images = np.concatenate(
            images_buffer,
            axis=0
        )

        shard_masks = np.concatenate(
            masks_buffer,
            axis=0
        )

        output_path = (
            output_dir
            / f"shard_{shard_id:03d}.npz"
        )

        np.savez(
            output_path,
            images=shard_images,
            masks=shard_masks
        )

        print(
            f"  Saved shard "
            f"{shard_id}: "
            f"{len(shard_images)} samples"
        )

    print()
    print(
        f"{split_name}: "
        f"{total_samples} samples"
    )


# ==================================================
# Run
# ==================================================

import sys


if len(sys.argv) != 2:
    print("Usage: uv run python src/preprocess_v2.py [train|val]")
    raise SystemExit(1)


split_name = sys.argv[1]

if split_name == "train":

    print("\nProcessing TRAIN...\n")

    preprocess_split(
        sampled["train"],
        "train"
    )

elif split_name == "val":

    print("\nProcessing VALIDATION...\n")

    preprocess_split(
        sampled["val"],
        "val"
    )

else:

    print(
        f"Unknown split: {split_name}"
    )

    raise SystemExit(1)


print("\nPreprocessing complete.")