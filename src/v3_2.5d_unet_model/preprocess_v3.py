from pathlib import Path
import json
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    / "v3"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v3"
)

TARGET_SIZE = 128

MODALITIES = [
    "t1",
    "t1ce",
    "t2",
    "flair",
]


# ============================================================
# MRI filenames
# ============================================================

MODALITY_FILES = {
    "t1": "t1n",
    "t1ce": "t1c",
    "t2": "t2w",
    "flair": "t2f",
}


# ============================================================
# Load patient volumes
# ============================================================

def load_patient(patient_id):

    patient_dir = DATASET_DIR / patient_id

    volumes = {}

    for modality in MODALITIES:

        suffix = MODALITY_FILES[modality]

        path = (
            patient_dir
            / f"{patient_id}-{suffix}.nii.gz"
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

        volumes[modality] = volume

    # --------------------------------------------------------
    # Normalize each modality
    # --------------------------------------------------------

    for modality in MODALITIES:

        volume = volumes[modality]

        nonzero = volume > 0

        if nonzero.any():

            mean = volume[nonzero].mean()
            std = volume[nonzero].std()

            if std > 0:

                volume[nonzero] = (
                    (volume[nonzero] - mean)
                    / std
                )

        volumes[modality] = volume

    # --------------------------------------------------------
    # Stack modalities
    # --------------------------------------------------------

    mri = np.stack(
        [
            volumes["t1"],
            volumes["t1ce"],
            volumes["t2"],
            volumes["flair"],
        ],
        axis=0,
    )

    # --------------------------------------------------------
    # Load segmentation
    # --------------------------------------------------------

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
    ).get_fdata()

    # Binary tumor mask
    seg = (
        seg > 0
    ).astype(np.float32)

    return mri, seg


# ============================================================
# Build 2.5D samples
# ============================================================

def preprocess_patient(
    patient_id,
    selected_slices,
):

    mri, seg = load_patient(
        patient_id
    )

    images = []
    masks = []

    scale = (
        TARGET_SIZE
        / mri.shape[-2]
    )

    for z in selected_slices:

        # ----------------------------------------------------
        # Neighboring slices
        # ----------------------------------------------------

        prev_z = max(
            z - 1,
            0
        )

        next_z = min(
            z + 1,
            mri.shape[-1] - 1
        )

        previous = mri[
            :,
            :,
            :,
            prev_z
        ]

        current = mri[
            :,
            :,
            :,
            z
        ]

        next_slice = mri[
            :,
            :,
            :,
            next_z
        ]

        # ----------------------------------------------------
        # 4 × 3 = 12 channels
        # ----------------------------------------------------

        image = np.concatenate(
            [
                previous,
                current,
                next_slice,
            ],
            axis=0,
        )

        # ----------------------------------------------------
        # Resize image
        # ----------------------------------------------------

        image = zoom(
            image,
            (
                1,
                scale,
                scale,
            ),
            order=1,
        )

        # ----------------------------------------------------
        # Target mask = center slice
        # ----------------------------------------------------

        mask = seg[
            :,
            :,
            z
        ]

        mask = zoom(
            mask,
            (
                scale,
                scale,
            ),
            order=0,
        )

        images.append(
            image.astype(
                np.float32
            )
        )

        masks.append(
            mask.astype(
                np.float32
            )
        )

    images = np.stack(
        images
    )

    masks = np.stack(
        masks
    )

    return images, masks


# ============================================================
# Save patient
# ============================================================

def save_patient(
    patient_id,
    selected_slices,
    output_dir,
):

    images, masks = preprocess_patient(
        patient_id,
        selected_slices,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{patient_id}.npz"
    )

    np.savez_compressed(
        output_path,
        images=images,
        masks=masks,
    )

    return (
        output_path,
        images.shape,
        masks.shape,
    )


# ============================================================
# Load metadata
# ============================================================

def load_metadata():

    with open(
        METADATA_DIR
        / "sampled_slices.json",
        "r",
    ) as f:

        sampled_slices = json.load(f)

    return sampled_slices


# ============================================================
# Process split
# ============================================================

def process_split(
    split,
    sampled_slices,
):

    output_dir = (
        OUTPUT_DIR
        / split
    )

    samples = sampled_slices[
        split
    ]

    # Group slices by patient
    patient_slices = {}

    for patient_id, slice_idx in samples:

        patient_slices.setdefault(
            patient_id,
            []
        ).append(
            slice_idx
        )

    total_patients = len(
        patient_slices
    )

    print()
    print("=" * 60)
    print(
        f"Processing {split.upper()} split"
    )
    print("=" * 60)

    print(
        "Patients:",
        total_patients
    )

    print(
        "Samples:",
        len(samples)
    )

    completed = 0
    skipped = 0
    failed = 0

    start_time = time.time()

    for patient_id, slices in patient_slices.items():

        output_path = (
            output_dir
            / f"{patient_id}.npz"
        )

        # ----------------------------------------------------
        # Skip already processed patients
        # ----------------------------------------------------

        if output_path.exists():

            skipped += 1

            continue

        try:

            patient_start = time.time()

            path, image_shape, mask_shape = (
                save_patient(
                    patient_id,
                    slices,
                    output_dir,
                )
            )

            elapsed = (
                time.time()
                - patient_start
            )

            completed += 1

            print(
                f"[{completed + skipped}/{total_patients}] "
                f"{patient_id} | "
                f"{len(slices)} slices | "
                f"{image_shape} | "
                f"{elapsed:.2f}s"
            )

        except Exception as e:

            failed += 1

            print(
                f"FAILED: {patient_id}"
            )

            print(
                "Error:",
                e
            )

    total_time = (
        time.time()
        - start_time
    )

    print()
    print("=" * 60)
    print(
        f"{split.upper()} preprocessing complete"
    )
    print("=" * 60)

    print(
        "Completed:",
        completed
    )

    print(
        "Skipped:",
        skipped
    )

    print(
        "Failed:",
        failed
    )

    print(
        "Total time:",
        f"{total_time / 60:.2f} minutes"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("V3 — 2.5D preprocessing")
    print("=" * 60)

    print(
        "Dataset:",
        DATASET_DIR
    )

    print(
        "Output:",
        OUTPUT_DIR
    )

    print(
        "Target size:",
        TARGET_SIZE
    )

    sampled_slices = load_metadata()

    process_split(
        "train",
        sampled_slices,
    )

    process_split(
        "val",
        sampled_slices,
    )


if __name__ == "__main__":
    main()
