from pathlib import Path
import json
import time

import numpy as np
import torch

from unet_3d import UNet3D
import preprocess_v4


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "v4"
    / "epoch_5.pt"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "v4_patient_split.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "v4"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Configuration
# ============================================================

THRESHOLD = 0.5

MAX_PATIENTS = 126


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("=" * 60)
print("V4 3D Brain Tumor Segmentation")
print("Mac Test Evaluation")
print("=" * 60)

print("Device:", device)


# ============================================================
# Validate paths
# ============================================================

print()
print("Raw data:", RAW_DIR)
print("Checkpoint:", CHECKPOINT)
print("Split:", SPLIT_FILE)

if not RAW_DIR.exists():
    raise FileNotFoundError(
        f"Raw data not found:\n{RAW_DIR}"
    )

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

if not SPLIT_FILE.exists():
    raise FileNotFoundError(
        f"Split file not found:\n{SPLIT_FILE}"
    )


# ============================================================
# Configure preprocessing
# ============================================================

preprocess_v4.RAW_DIR = RAW_DIR

print()
print("Patch size:", preprocess_v4.PATCH_SIZE)
print("Stride:", preprocess_v4.STRIDE)


# ============================================================
# Load test split
# ============================================================

with open(SPLIT_FILE) as f:
    split = json.load(f)

test_patients = split["test"]

test_patients = test_patients[:MAX_PATIENTS]

print()
print("Test patients selected:", len(test_patients))


# ============================================================
# Load model
# ============================================================

model = UNet3D(
    in_channels=4,
    out_channels=1
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("Checkpoint loaded.")


# ============================================================
# Normalize MRI
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
# Patch positions
# ============================================================

def get_patch_positions(shape):

    _, depth, height, width = shape

    pd, ph, pw = preprocess_v4.PATCH_SIZE

    sd, sh, sw = preprocess_v4.STRIDE

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
# Dice
# ============================================================

def dice_score(prediction, target):

    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)

    intersection = (
        prediction * target
    ).sum()

    return (
        2.0 * intersection + 1e-6
    ) / (
        prediction.sum()
        + target.sum()
        + 1e-6
    )


# ============================================================
# IoU
# ============================================================

def iou_score(prediction, target):

    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)

    intersection = (
        prediction * target
    ).sum()

    union = (
        prediction
        + target
        - prediction * target
    ).sum()

    return (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )


# ============================================================
# Validate one patient
# ============================================================

@torch.no_grad()
def validate_patient(patient_id):

    start = time.time()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    mri, target = (
        preprocess_v4.load_patient(
            patient_id
        )
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    mri = normalize_mri(mri)

    _, depth, height, width = mri.shape

    pd, ph, pw = preprocess_v4.PATCH_SIZE

    # --------------------------------------------------------
    # Accumulation buffers
    # --------------------------------------------------------

    probability_sum = np.zeros(
        (
            depth,
            height,
            width
        ),
        dtype=np.float32
    )

    count_map = np.zeros(
        (
            depth,
            height,
            width
        ),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Patch positions
    # --------------------------------------------------------

    z_starts, y_starts, x_starts = (
        get_patch_positions(
            mri.shape
        )
    )

    patch_count = 0

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    for z in z_starts:

        for y in y_starts:

            for x in x_starts:

                patch = mri[
                    :,
                    z:z + pd,
                    y:y + ph,
                    x:x + pw
                ]

                tensor = torch.from_numpy(
                    patch.astype(np.float32)
                ).unsqueeze(0).to(device)

                logits = model(tensor)

                probability = (
                    torch.sigmoid(logits)
                    .squeeze()
                    .detach()
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

    # --------------------------------------------------------
    # Reconstruct
    # --------------------------------------------------------

    probability_volume = (
        probability_sum
        / np.maximum(
            count_map,
            1
        )
    )

    prediction = (
        probability_volume >= THRESHOLD
    ).astype(np.uint8)

    target_binary = (
        target > 0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    dice = dice_score(
        prediction,
        target_binary
    )

    iou = iou_score(
        prediction,
        target_binary
    )

    elapsed = time.time() - start

    return {
        "patient": patient_id,
        "dice": float(dice),
        "iou": float(iou),
        "patches": patch_count,
        "time_seconds": float(elapsed)
    }


# ============================================================
# Evaluation
# ============================================================

results = []

for index, patient_id in enumerate(
    test_patients,
    start=1
):

    print()
    print(
        f"Patient {index}/{len(test_patients)}: "
        f"{patient_id}"
    )

    try:

        result = validate_patient(
            patient_id
        )

        results.append(result)

        print(
            f"  Patches: {result['patches']}"
        )

        print(
            f"  Dice: {result['dice']:.4f}"
        )

        print(
            f"  IoU:  {result['iou']:.4f}"
        )

        print(
            f"  Time: {result['time_seconds']:.2f}s"
        )

    except Exception as e:

        print(
            f"  FAILED: {e}"
        )


# ============================================================
# Save results
# ============================================================

if not results:

    raise RuntimeError(
        "No patients were successfully evaluated."
    )


dice_values = np.array([
    r["dice"]
    for r in results
])

iou_values = np.array([
    r["iou"]
    for r in results
])


summary = {

    "checkpoint": str(CHECKPOINT),

    "patients_evaluated":
        len(results),

    "mean_dice":
        float(dice_values.mean()),

    "std_dice":
        float(dice_values.std()),

    "mean_iou":
        float(iou_values.mean()),

    "std_iou":
        float(iou_values.std()),

    "best_dice":
        float(dice_values.max()),

    "worst_dice":
        float(dice_values.min()),

    "results":
        results
}


OUTPUT_FILE = (
    RESULTS_DIR
    / "v4_test_5_patients.json"
)

with open(
    OUTPUT_FILE,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# Final output
# ============================================================

print()
print("=" * 60)
print("V4 TEST EVALUATION")
print("=" * 60)

print(
    "Patients:",
    len(results)
)

print(
    f"Mean Dice: "
    f"{dice_values.mean():.4f}"
)

print(
    f"Std Dice:  "
    f"{dice_values.std():.4f}"
)

print(
    f"Mean IoU:  "
    f"{iou_values.mean():.4f}"
)

print(
    f"Std IoU:   "
    f"{iou_values.std():.4f}"
)

print(
    f"Best Dice: "
    f"{dice_values.max():.4f}"
)

print(
    f"Worst Dice: "
    f"{dice_values.min():.4f}"
)

print()
print(
    "Saved:",
    OUTPUT_FILE
)