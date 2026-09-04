from pathlib import Path
import argparse
import time

import nibabel as nib
import numpy as np
import torch

from v4_3d_unet_model.unet_3d import UNet3D
from v4_3d_unet_model import preprocess_v4


# ============================================================
# Configuration
# ============================================================

PATCH_SIZE = preprocess_v4.PATCH_SIZE
STRIDE = preprocess_v4.STRIDE

THRESHOLD = 0.5


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# Model
# ============================================================

def load_model(checkpoint_path):

    model = UNet3D(
        in_channels=4,
        out_channels=1
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# MRI Loading
# ============================================================

def load_mri(patient_dir):

    patient_dir = Path(patient_dir)

    patient_id = patient_dir.name

    volumes = []

    for modality in preprocess_v4.MODALITIES:

        path = (
            patient_dir
            / f"{patient_id}-{modality}.nii.gz"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing modality:\n{path}"
            )

        volume = nib.load(
            path
        ).get_fdata(
            dtype=np.float32
        )

        volumes.append(volume)

    # (C, H, W, Z)
    mri = np.stack(
        volumes,
        axis=0
    )

    # (C, Z, H, W)
    mri = np.transpose(
        mri,
        (0, 3, 1, 2)
    )

    return mri


# ============================================================
# Normalization
# ============================================================

def normalize_mri(mri):

    mri = mri.copy()

    for channel in range(mri.shape[0]):

        mri[channel] = (
            preprocess_v4.normalize_volume(
                mri[channel]
            )
        )

    return mri


# ============================================================
# Patch Positions
# ============================================================

def get_patch_positions(shape):

    _, depth, height, width = shape

    pd, ph, pw = PATCH_SIZE
    sd, sh, sw = STRIDE

    z_starts = preprocess_v4.get_patch_starts(
        depth,
        pd,
        sd
    )

    y_starts = preprocess_v4.get_patch_starts(
        height,
        ph,
        sh
    )

    x_starts = preprocess_v4.get_patch_starts(
        width,
        pw,
        sw
    )

    return (
        z_starts,
        y_starts,
        x_starts
    )


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict_volume(model, mri):

    mri = normalize_mri(mri)

    _, depth, height, width = mri.shape

    pd, ph, pw = PATCH_SIZE

    probability_sum = np.zeros(
        (depth, height, width),
        dtype=np.float32
    )

    count_map = np.zeros(
        (depth, height, width),
        dtype=np.float32
    )

    z_starts, y_starts, x_starts = (
        get_patch_positions(mri.shape)
    )

    patch_count = 0

    for z in z_starts:

        for y in y_starts:

            for x in x_starts:

                patch = mri[
                    :,
                    z:z + pd,
                    y:y + ph,
                    x:x + pw
                ]

                tensor = (
                    torch.from_numpy(
                        patch.astype(np.float32)
                    )
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                logits = model(tensor)

                probability = (
                    torch.sigmoid(logits)
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                probability_sum[
                    z:z + pd,
                    y:y + ph,
                    x:x + pw
                ] += probability

                count_map[
                    z:z + pd,
                    y:y + ph,
                    x:x + pw
                ] += 1

                patch_count += 1

    probability_volume = (
        probability_sum
        / np.maximum(count_map, 1)
    )

    prediction = (
        probability_volume >= THRESHOLD
    ).astype(np.uint8)

    return (
        prediction,
        probability_volume,
        patch_count
    )


# ============================================================
# Save NIfTI
# ============================================================

def save_prediction(
    prediction,
    reference_path,
    output_path
):

    reference = nib.load(
        reference_path
    )

    # Prediction is (Z, H, W)
    # NIfTI expects (H, W, Z)
    prediction_nifti = np.transpose(
        prediction,
        (1, 2, 0)
    )

    image = nib.Nifti1Image(
        prediction_nifti.astype(np.uint8),
        affine=reference.affine,
        header=reference.header
    )

    nib.save(
        image,
        output_path
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "V4 3D Brain Tumor MRI Segmentation"
        )
    )

    parser.add_argument(
        "--patient_dir",
        required=True,
        help="Path to BraTS patient directory"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to V4 checkpoint"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output prediction NIfTI path"
    )

    args = parser.parse_args()

    patient_dir = Path(
        args.patient_dir
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("V4 3D BRAIN TUMOR SEGMENTATION")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Patient:", patient_dir)
    print("Checkpoint:", checkpoint_path)

    start = time.time()

    model = load_model(
        checkpoint_path
    )

    mri = load_mri(
        patient_dir
    )

    prediction, probability, patches = (
        predict_volume(
            model,
            mri
        )
    )

    # Reference T1 volume for affine/header
    patient_id = patient_dir.name

    reference_path = (
        patient_dir
        / f"{patient_id}-t1n.nii.gz"
    )

    save_prediction(
        prediction,
        reference_path,
        output_path
    )

    elapsed = time.time() - start

    print()
    print("Prediction complete.")
    print("Patches:", patches)
    print(
        "Predicted tumor voxels:",
        int(prediction.sum())
    )
    print(
        "Output:",
        output_path
    )
    print(
        f"Time: {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()