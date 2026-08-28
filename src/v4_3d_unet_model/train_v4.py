from pathlib import Path
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


# ==================================================
# Paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

V4_DIR = (
    PROJECT_ROOT
    / "src"
    / "v4_3d_unet_model"
)

sys.path.insert(
    0,
    str(V4_DIR)
)

from dataset import BraTS3DDataset
from sampler import BalancedPatchSampler
from unet_3d import UNet3D


# ==================================================
# Configuration
# ==================================================

BATCH_SIZE = 1

NUM_WORKERS = 0

EPOCHS = 10

SAMPLES_PER_EPOCH = 4000

TUMOR_RATIO = 0.5

LEARNING_RATE = 1e-4

THRESHOLD = 0.5

SEED = 42


# ==================================================
# Checkpoint directory
# ==================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "v4"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# Reproducibility
# ==================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# ==================================================
# Device
# ==================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")


print("=" * 60)
print("V4 3D Brain Tumor Segmentation")
print("=" * 60)

print(
    "Device:",
    device
)


# ==================================================
# Dataset paths
# ==================================================

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "train"
)

VAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "val"
)

TRAIN_INDEX = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v4"
    / "metadata"
    / "train_patch_index.json"
)


# ==================================================
# Dataset
# ==================================================

print()
print("Loading datasets...")

train_dataset = BraTS3DDataset(
    TRAIN_DIR
)

val_dataset = BraTS3DDataset(
    VAL_DIR
)

print(
    "Training patches:",
    len(train_dataset)
)

print(
    "Validation patches:",
    len(val_dataset)
)


# ==================================================
# Balanced sampler
# ==================================================

sampler = BalancedPatchSampler(
    dataset=train_dataset,
    index_path=TRAIN_INDEX,
    samples_per_epoch=SAMPLES_PER_EPOCH,
    tumor_ratio=TUMOR_RATIO,
    seed=SEED
)


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
# Model
# ==================================================

model = UNet3D(
    in_channels=4,
    out_channels=1
).to(device)


print()
print(
    "Model:",
    model.__class__.__name__
)


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
# Training
# ==================================================

for epoch in range(EPOCHS):

    epoch_start = time.time()

    print()
    print("=" * 60)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}]"
    )

    print("=" * 60)

    # ==================================================
    # Create balanced training subset
    # ==================================================

    sampled_indices = (
        sampler.sample_indices()
    )

    train_subset = Subset(
        train_dataset,
        sampled_indices
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # ==================================================
    # Training
    # ==================================================

    model.train()

    train_loss = 0.0

    train_dice = 0.0

    train_iou = 0.0

    train_batches = 0

    train_start = time.time()

    for images, masks in train_loader:

        images = images.to(
            device
        )

        masks = masks.to(
            device
        )

        masks = masks.unsqueeze(1)

        optimizer.zero_grad()

        logits = model(
            images
        )

        loss = combined_loss(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

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

        if train_batches % 500 == 0:

            print(
                f"  Training batch "
                f"{train_batches}/"
                f"{len(train_loader)}"
            )

    train_loss /= train_batches

    train_dice /= train_batches

    train_iou /= train_batches

    train_time = (
        time.time()
        - train_start
    )

    # ==================================================
    # Validation
    # ==================================================

    model.eval()

    val_loss = 0.0

    val_dice = 0.0

    val_iou = 0.0

    val_batches = 0

    val_start = time.time()

    # --------------------------------------------------
    # IMPORTANT:
    # Use a subset for validation during training.
    #
    # Full validation is done separately after training.
    # --------------------------------------------------

    val_indices = list(
        range(
            min(
                500,
                len(val_dataset)
            )
        )
    )

    val_subset = Subset(
        val_dataset,
        val_indices
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                device
            )

            masks = masks.to(
                device
            )

            masks = masks.unsqueeze(1)

            logits = model(
                images
            )

            loss = combined_loss(
                logits,
                masks
            )

            val_loss += loss.item()

            val_dice += dice_score(
                logits,
                masks,
                THRESHOLD
            )

            val_iou += iou_score(
                logits,
                masks,
                THRESHOLD
            )

            val_batches += 1

    val_loss /= val_batches

    val_dice /= val_batches

    val_iou /= val_batches

    val_time = (
        time.time()
        - val_start
    )

    epoch_time = (
        time.time()
        - epoch_start
    )

    # ==================================================
    # History
    # ==================================================

    epoch_result = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_dice": train_dice,
        "train_iou": train_iou,
        "val_loss": val_loss,
        "val_dice": val_dice,
        "val_iou": val_iou,
        "train_time": train_time,
        "val_time": val_time,
        "epoch_time": epoch_time
    }

    history.append(
        epoch_result
    )

    # ==================================================
    # Print results
    # ==================================================

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
        f"Val Loss:   {val_loss:.4f}"
    )

    print(
        f"Val Dice:   {val_dice:.4f}"
    )

    print(
        f"Val IoU:    {val_iou:.4f}"
    )

    print(
        f"Train time: {train_time / 60:.2f} min"
    )

    print(
        f"Val time:   {val_time / 60:.2f} min"
    )

    print(
        f"Epoch time: {epoch_time / 60:.2f} min"
    )

    # ==================================================
    # Checkpoint
    # ==================================================

    checkpoint = {

        "epoch":
            epoch + 1,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "train_dice":
            train_dice,

        "train_iou":
            train_iou,

        "val_loss":
            val_loss,

        "val_dice":
            val_dice,

        "val_iou":
            val_iou,

        "history":
            history,

        "config": {

            "batch_size":
                BATCH_SIZE,

            "samples_per_epoch":
                SAMPLES_PER_EPOCH,

            "tumor_ratio":
                TUMOR_RATIO,

            "learning_rate":
                LEARNING_RATE,

            "patch_size":
                [64, 128, 128],

            "seed":
                SEED
        }
    }

    torch.save(
        checkpoint,
        CHECKPOINT_DIR
        / "latest.pt"
    )

    # ==================================================
    # Best model
    # ==================================================

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        torch.save(
            checkpoint,
            CHECKPOINT_DIR
            / "best.pt"
        )

        print()
        print(
            "New best model!"
        )

        print(
            f"Best Val Dice: "
            f"{best_val_dice:.4f}"
        )


# ==================================================
# Save history
# ==================================================

history_path = (
    CHECKPOINT_DIR
    / "history.json"
)

with open(
    history_path,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )


# ==================================================
# Finished
# ==================================================

print()
print("=" * 60)
print("V4 TRAINING COMPLETE")
print("=" * 60)

print(
    "Best validation Dice:",
    f"{best_val_dice:.4f}"
)

print(
    "Best checkpoint:",
    CHECKPOINT_DIR
    / "best.pt"
)

print(
    "Latest checkpoint:",
    CHECKPOINT_DIR
    / "latest.pt"
)

print(
    "History:",
    history_path
)