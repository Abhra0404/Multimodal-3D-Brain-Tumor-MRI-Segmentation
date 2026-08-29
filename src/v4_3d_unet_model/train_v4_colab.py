from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from unet_3d import UNet3D
import preprocess_v4


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = Path(
    "/content/drive/MyDrive/"
    "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "v4"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 2

SAMPLES_PER_EPOCH = 4000

TUMOR_RATIO = 0.5

EPOCHS = 3

LEARNING_RATE = 1e-4

THRESHOLD = 0.5

SEED = 42

PATIENTS_PER_CYCLE = 20


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("V4 3D Brain Tumor Segmentation - Colab")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# IMPORTANT:
# Point preprocess_v4 to Google Drive
# ============================================================

preprocess_v4.RAW_DIR = RAW_DIR


# ============================================================
# Patient discovery
# ============================================================

patients = sorted(
    p.name
    for p in RAW_DIR.iterdir()
    if p.is_dir()
)

print(
    "Patients:",
    len(patients)
)


# ============================================================
# Loss
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(
    logits,
    targets,
    smooth=1e-6
):

    probabilities = torch.sigmoid(
        logits
    )

    probabilities = probabilities.flatten(
        1
    )

    targets = targets.flatten(
        1
    )

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        probabilities.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return 1.0 - dice.mean()


def combined_loss(
    logits,
    targets
):

    return (
        bce_loss(
            logits,
            targets
        )
        + dice_loss(
            logits,
            targets
        )
    )


# ============================================================
# Metrics
# ============================================================

def dice_score(
    logits,
    targets,
    threshold=0.5,
    smooth=1e-6
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities > threshold
    ).float()

    predictions = predictions.flatten(
        1
    )

    targets = targets.flatten(
        1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return dice.mean().item()


def iou_score(
    logits,
    targets,
    threshold=0.5,
    smooth=1e-6
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities > threshold
    ).float()

    predictions = predictions.flatten(
        1
    )

    targets = targets.flatten(
        1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum(dim=1)

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean().item()


# ============================================================
# Load + preprocess ONE patient
# ============================================================

def prepare_patient(
    patient_id
):

    mri, mask = (
        preprocess_v4.load_patient(
            patient_id
        )
    )

    patches = extract_patient_patches(
        mri,
        mask
    )

    tumor = [
        p for p in patches
        if p["tumor_voxels"] > 0
    ]

    background = [
        p for p in patches
        if p["tumor_voxels"] == 0
    ]

    return tumor, background


# ============================================================
# Balanced batch sampling
# ============================================================

def make_batch(
    patient_cache,
    batch_size,
    tumor_probability=0.5
):

    images = []
    masks = []

    patient_ids = list(
        patient_cache.keys()
    )

    for _ in range(batch_size):

        patient_id = random.choice(
            patient_ids
        )

        data = patient_cache[
            patient_id
        ]

        use_tumor = (
            random.random()
            < tumor_probability
        )

        if (
            use_tumor
            and data["tumor"]
        ):

            patch = random.choice(
                data["tumor"]
            )

        elif data["background"]:

            patch = random.choice(
                data["background"]
            )

        else:

            patch = random.choice(
                data["tumor"]
            )

        images.append(
            torch.from_numpy(
                patch["image"]
            )
        )

        masks.append(
            torch.from_numpy(
                patch["mask"]
            )
        )

    images = torch.stack(
        images
    ).float()

    masks = torch.stack(
        masks
    ).float()

    masks = masks.unsqueeze(1)

    return images, masks


# ============================================================
# Model
# ============================================================

model = UNet3D(
    in_channels=4,
    out_channels=1
).to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# AMP
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# ============================================================
# Training
# ============================================================

history = []

best_val_dice = 0.0


for epoch in range(EPOCHS):

    epoch_start = time.time()

    print()
    print("=" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Select patients for this cycle
    # --------------------------------------------------------

    random.shuffle(
        patients
    )

    selected_patients = patients[
        :PATIENTS_PER_CYCLE
    ]

    # --------------------------------------------------------
    # Build patient cache
    # --------------------------------------------------------

    patient_cache = {}

    cache_start = time.time()

    for i, patient_id in enumerate(
        selected_patients
    ):

        tumor, background = (
            prepare_patient(
                patient_id
            )
        )

        # Skip pathological cases
        if not tumor and not background:
            continue

        patient_cache[
            patient_id
        ] = {
            "tumor": tumor,
            "background": background
        }

        print(
            f"Loaded {i + 1}/"
            f"{PATIENTS_PER_CYCLE}: "
            f"{patient_id}"
        )

    cache_time = (
        time.time()
        - cache_start
    )

    print(
        f"Cache time: "
        f"{cache_time / 60:.2f} min"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_dice = 0.0
    train_iou = 0.0

    train_batches = 0

    train_start = time.time()

    total_batches = (
        SAMPLES_PER_EPOCH
        // BATCH_SIZE
    )

    for batch_idx in range(
        total_batches
    ):

        images, masks = make_batch(
            patient_cache,
            BATCH_SIZE,
            TUMOR_RATIO
        )

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):

            logits = model(
                images
            )

            loss = combined_loss(
                logits,
                masks
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        train_loss += (
            loss.item()
        )

        train_dice += dice_score(
            logits,
            masks,
            THRESHOLD
        )

        train_iou += iou_score(
            logits,
            masks,
            THRESHOLD
        )

        train_batches += 1

        if (
            (batch_idx + 1) % 500
            == 0
        ):

            print(
                f"Training: "
                f"{batch_idx + 1}/"
                f"{total_batches}"
            )

    train_loss /= train_batches
    train_dice /= train_batches
    train_iou /= train_batches

    train_time = (
        time.time()
        - train_start
    )

    # --------------------------------------------------------
    # Cleanup patient cache
    # --------------------------------------------------------

    del patient_cache

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    checkpoint = {

        "epoch":
            epoch + 1,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "train_loss":
            train_loss,

        "train_dice":
            train_dice,

        "train_iou":
            train_iou,

        "config": {

            "batch_size":
                BATCH_SIZE,

            "samples_per_epoch":
                SAMPLES_PER_EPOCH,

            "tumor_ratio":
                TUMOR_RATIO,

            "learning_rate":
                LEARNING_RATE,

            "epochs":
                EPOCHS,

            "seed":
                SEED
        }
    }

    torch.save(
        checkpoint,
        CHECKPOINT_DIR
        / f"epoch_{epoch + 1}.pt"
    )

    torch.save(
        checkpoint,
        CHECKPOINT_DIR
        / "latest.pt"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    epoch_time = (
        time.time()
        - epoch_start
    )

    result = {

        "epoch":
            epoch + 1,

        "train_loss":
            train_loss,

        "train_dice":
            train_dice,

        "train_iou":
            train_iou,

        "train_time":
            train_time,

        "cache_time":
            cache_time,

        "epoch_time":
            epoch_time
    }

    history.append(
        result
    )

    print()
    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Dice: {train_dice:.4f}"
    )

    print(
        f"Train IoU:  {train_iou:.4f}"
    )

    print(
        f"Cache time: "
        f"{cache_time / 60:.2f} min"
    )

    print(
        f"Train time: "
        f"{train_time / 60:.2f} min"
    )

    print(
        f"Epoch time: "
        f"{epoch_time / 60:.2f} min"
    )


# ============================================================
# Save history
# ============================================================

with open(
    CHECKPOINT_DIR
    / "history.json",
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )


print()
print("=" * 60)
print("V4 TRAINING COMPLETE")
print("=" * 60)