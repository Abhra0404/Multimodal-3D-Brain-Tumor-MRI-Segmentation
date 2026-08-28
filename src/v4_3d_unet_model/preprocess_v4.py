from pathlib import Path
import json
import time

import nibabel as nib
import numpy as np


# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
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

PATIENT_SPLIT_PATH = (
    METADATA_DIR
    / "patient_split.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
)


# ==================================================
# Configuration
# ==================================================

MODALITIES = [
    "t1n",
    "t1c",
    "t2w",
    "t2f"
]

# (Depth, Height, Width)
PATCH_SIZE = (
    64,
    128,
    128
)

# (Depth, Height, Width)
STRIDE = (
    32,
    64,
    64
)


# ==================================================
# Load Patient Split
# ==================================================

def load_patient_split():

    if not PATIENT_SPLIT_PATH.exists():

        raise FileNotFoundError(
            f"Patient split not found:\n"
            f"{PATIENT_SPLIT_PATH}"
        )

    with open(
        PATIENT_SPLIT_PATH,
        "r"
    ) as f:

        split = json.load(f)

    train_ids = split["train_ids"]
    val_ids = split["val_ids"]

    print(
        "Training patients:",
        len(train_ids)
    )

    print(
        "Validation patients:",
        len(val_ids)
    )

    return train_ids, val_ids


# ==================================================
# Load Patient
# ==================================================

def load_patient(patient_id):

    patient_dir = (
        RAW_DIR
        / patient_id
    )

    volumes = []

    for modality in MODALITIES:

        path = (
            patient_dir
            / f"{patient_id}-{modality}.nii.gz"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing modality file:\n{path}"
            )

        volume = nib.load(
            path
        ).get_fdata(
            dtype=np.float32
        )

        volumes.append(volume)

    # ----------------------------------------------
    # Segmentation
    # ----------------------------------------------

    seg_path = (
        patient_dir
        / f"{patient_id}-seg.nii.gz"
    )

    if not seg_path.exists():

        raise FileNotFoundError(
            f"Missing segmentation file:\n{seg_path}"
        )

    seg = nib.load(
        seg_path
    ).get_fdata(
        dtype=np.float32
    )

    # ----------------------------------------------
    # MRI
    #
    # Original:
    # (H, W, Z)
    #
    # Stack:
    # (C, H, W, Z)
    #
    # Convert:
    # (C, Z, H, W)
    # ----------------------------------------------

    mri = np.stack(
        volumes,
        axis=0
    )

    mri = np.transpose(
        mri,
        (0, 3, 1, 2)
    )

    # ----------------------------------------------
    # Mask
    #
    # Original:
    # (H, W, Z)
    #
    # Convert:
    # (Z, H, W)
    # ----------------------------------------------

    seg = np.transpose(
        seg,
        (2, 0, 1)
    )

    # ----------------------------------------------
    # Binary tumor mask
    # ----------------------------------------------

    seg = (
        seg > 0
    ).astype(
        np.float32
    )

    return mri, seg


# ==================================================
# Normalize
# ==================================================

def normalize_volume(volume):

    volume = volume.astype(
        np.float32
    )

    nonzero = (
        volume != 0
    )

    if not np.any(nonzero):

        return volume

    mean = volume[
        nonzero
    ].mean()

    std = volume[
        nonzero
    ].std()

    if std < 1e-6:

        return volume

    volume[nonzero] = (
        volume[nonzero] - mean
    ) / std

    return volume


# ==================================================
# Patch Positions
# ==================================================

def get_patch_starts(
    size,
    patch_size,
    stride
):

    if size <= patch_size:

        return [0]

    starts = list(
        range(
            0,
            size - patch_size + 1,
            stride
        )
    )

    final_start = (
        size - patch_size
    )

    if starts[-1] != final_start:

        starts.append(
            final_start
        )

    return starts


# ==================================================
# Process One Patient
# ==================================================

def preprocess_patient(
    patient_id,
    output_dir
):

    mri, mask = load_patient(
        patient_id
    )

    # ----------------------------------------------
    # Normalize each modality
    # ----------------------------------------------

    for channel in range(
        mri.shape[0]
    ):

        mri[channel] = normalize_volume(
            mri[channel]
        )

    _, depth, height, width = (
        mri.shape
    )

    patch_depth, patch_height, patch_width = (
        PATCH_SIZE
    )

    stride_depth, stride_height, stride_width = (
        STRIDE
    )

    # ----------------------------------------------
    # Patch positions
    # ----------------------------------------------

    z_starts = get_patch_starts(
        depth,
        patch_depth,
        stride_depth
    )

    y_starts = get_patch_starts(
        height,
        patch_height,
        stride_height
    )

    x_starts = get_patch_starts(
        width,
        patch_width,
        stride_width
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    patch_count = 0
    tumor_patches = 0
    background_patches = 0

    # ----------------------------------------------
    # Extract patches
    # ----------------------------------------------

    for z in z_starts:

        for y in y_starts:

            for x in x_starts:

                image_patch = mri[
                    :,
                    z:z + patch_depth,
                    y:y + patch_height,
                    x:x + patch_width
                ]

                mask_patch = mask[
                    z:z + patch_depth,
                    y:y + patch_height,
                    x:x + patch_width
                ]

                tumor_voxels = int(
                    mask_patch.sum()
                )

                if tumor_voxels > 0:

                    tumor_patches += 1

                else:

                    background_patches += 1

                patch_path = (
                    output_dir
                    / (
                        f"{patient_id}"
                        f"_z{z}"
                        f"_y{y}"
                        f"_x{x}.npz"
                    )
                )

                np.savez_compressed(
                    patch_path,
                    image=image_patch.astype(
                        np.float32
                    ),
                    mask=mask_patch.astype(
                        np.float32
                    ),
                    patient_id=patient_id,
                    z=z,
                    y=y,
                    x=x
                )

                patch_count += 1

    return {
        "patches": patch_count,
        "tumor_patches": tumor_patches,
        "background_patches": background_patches
    }


# ==================================================
# Process Split
# ==================================================

def preprocess_split(
    patient_ids,
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

    completed = 0
    failed = 0

    total_patches = 0
    total_tumor_patches = 0
    total_background_patches = 0

    start_time = time.time()

    print()
    print("=" * 60)
    print(
        f"{split_name.upper()} V4 PREPROCESSING"
    )
    print("=" * 60)

    print(
        "Patients:",
        len(patient_ids)
    )

    for index, patient_id in enumerate(
        patient_ids,
        start=1
    ):

        try:

            result = preprocess_patient(
                patient_id,
                output_dir
            )

            completed += 1

            total_patches += (
                result["patches"]
            )

            total_tumor_patches += (
                result["tumor_patches"]
            )

            total_background_patches += (
                result["background_patches"]
            )

            print(
                f"[{index}/{len(patient_ids)}]"
                f" {patient_id}"
                f" → "
                f"{result['patches']} patches"
                f" | tumor:"
                f" {result['tumor_patches']}"
                f" | background:"
                f" {result['background_patches']}"
            )

        except Exception as e:

            failed += 1

            print(
                f"[{index}/{len(patient_ids)}]"
                f" FAILED:"
                f" {patient_id}"
            )

            print(
                " ",
                str(e)
            )

    elapsed = (
        time.time()
        - start_time
    )

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    print()
    print("=" * 60)
    print(
        f"{split_name.upper()} PREPROCESSING COMPLETE"
    )
    print("=" * 60)

    print(
        "Completed:",
        completed
    )

    print(
        "Failed:",
        failed
    )

    print(
        "Total patches:",
        total_patches
    )

    print(
        "Tumor patches:",
        total_tumor_patches
    )

    print(
        "Background patches:",
        total_background_patches
    )

    print(
        f"Time: {elapsed / 60:.2f} minutes"
    )

    print(
        "Output:",
        output_dir
    )


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    train_ids, val_ids = (
        load_patient_split()
    )

    preprocess_split(
        train_ids,
        "train"
    )

    preprocess_split(
        val_ids,
        "val"
    )