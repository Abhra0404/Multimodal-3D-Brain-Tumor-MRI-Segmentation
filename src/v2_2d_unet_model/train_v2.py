from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT / "src")
)

from v2_2d_unet_model.dataset import BraTSSliceDataset
from v2_2d_unet_model.unet import UNet


# ==================================================
# Configuration
# ==================================================

BATCH_SIZE = 8
NUM_WORKERS = 0

EPOCHS = 10
LEARNING_RATE = 1e-4

THRESHOLD = 0.5

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "v2"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# Device
# ==================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")


print("=" * 60)
print("V2 Brain Tumor Segmentation")
print("=" * 60)

print("Device:", device)


# ==================================================
# Dataset paths
# ==================================================

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v2"
    / "train"
)

VAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v2"
    / "val"
)


# ==================================================
# Dataset
# ==================================================

train_dataset = BraTSSliceDataset(
    TRAIN_DIR
)

val_dataset = BraTSSliceDataset(
    VAL_DIR
)

print()
print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ==================================================
# DataLoader
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)


# ==================================================
# Model
# ==================================================

model = UNet(
    in_channels=4,
    out_channels=1
).to(device)


# ==================================================
# Loss functions
# ==================================================

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


# ==================================================
# Metrics
# ==================================================

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


# ==================================================
# Optimizer
# ==================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# Training state
# ==================================================

best_val_dice = 0.0

history = []


# ==================================================
# Training loop
# ==================================================

for epoch in range(EPOCHS):

    epoch_start = time.time()

    # ----------------------------------------------
    # Training
    # ----------------------------------------------

    model.train()

    train_loss = 0.0
    train_dice = 0.0

    for images, masks in train_loader:

        images = images.to(device)

        masks = masks.to(device)

        masks = masks.unsqueeze(1)

        optimizer.zero_grad()

        logits = model(images)

        loss = combined_loss(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        train_dice += dice_score(
            logits,
            masks
        )

    train_loss /= len(
        train_loader
    )

    train_dice /= len(
        train_loader
    )

    # ----------------------------------------------
    # Validation
    # ----------------------------------------------

    model.eval()

    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)

            masks = masks.to(device)

            masks = masks.unsqueeze(1)

            logits = model(images)

            loss = combined_loss(
                logits,
                masks
            )

            val_loss += loss.item()

            val_dice += dice_score(
                logits,
                masks
            )

            val_iou += iou_score(
                logits,
                masks
            )

    val_loss /= len(
        val_loader
    )

    val_dice /= len(
        val_loader
    )

    val_iou /= len(
        val_loader
    )

    epoch_time = (
        time.time()
        - epoch_start
    )

    # ----------------------------------------------
    # Save history
    # ----------------------------------------------

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_dice": train_dice,
        "val_loss": val_loss,
        "val_dice": val_dice,
        "val_iou": val_iou,
        "time": epoch_time,
    })

    # ----------------------------------------------
    # Print results
    # ----------------------------------------------

    print()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| "
        f"Train Loss: {train_loss:.4f} "
        f"| "
        f"Train Dice: {train_dice:.4f} "
        f"| "
        f"Val Loss: {val_loss:.4f} "
        f"| "
        f"Val Dice: {val_dice:.4f} "
        f"| "
        f"Val IoU: {val_iou:.4f} "
        f"| "
        f"Time: {epoch_time / 60:.2f} min"
    )

    # ----------------------------------------------
    # Checkpoint
    # ----------------------------------------------

    checkpoint = {
        "epoch": epoch + 1,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "train_dice":
            train_dice,

        "val_loss":
            val_loss,

        "val_dice":
            val_dice,

        "val_iou":
            val_iou,

        "history":
            history,
    }

    torch.save(
        checkpoint,
        CHECKPOINT_DIR
        / "latest.pt"
    )

    # ----------------------------------------------
    # Best model
    # ----------------------------------------------

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        torch.save(
            checkpoint,
            CHECKPOINT_DIR
            / "best.pt"
        )

        print(
            f"  New best model "
            f"| Val Dice: {val_dice:.4f}"
        )


# ==================================================
# Finished
# ==================================================

print()
print("=" * 60)
print("Training complete")
print("=" * 60)

print(
    "Best validation Dice:",
    f"{best_val_dice:.4f}"
)

print(
    "Best checkpoint:",
    CHECKPOINT_DIR / "best.pt"
)

print(
    "Latest checkpoint:",
    CHECKPOINT_DIR / "latest.pt"
)