import time
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from v2_2d_unet_model.dataset import BraTSSliceDataset
from v2_2d_unet_model.unet import UNet


# ==================================================
# Configuration
# ==================================================

BATCH_SIZE = 8
NUM_BENCHMARK_BATCHES = 100


# ==================================================
# Device
# ==================================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ==================================================
# Dataset
# ==================================================

dataset = BraTSSliceDataset(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v2"
    / "train"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Samples:", len(dataset))
print("Batch size:", BATCH_SIZE)
print("Batches per epoch:", len(loader))


# ==================================================
# Model
# ==================================================

model = UNet().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


# ==================================================
# Benchmark
# ==================================================

print(
    f"\nRunning "
    f"{NUM_BENCHMARK_BATCHES} training batches..."
)

model.train()

start = time.time()

for i, (images, masks) in enumerate(loader):

    images = images.to(device)

    masks = masks.to(device)

    masks = masks.unsqueeze(1)

    optimizer.zero_grad()

    logits = model(images)

    # Dummy loss for throughput benchmark only.
    # This is NOT used for actual training.
    loss = logits.mean()

    loss.backward()

    optimizer.step()

    if i + 1 >= NUM_BENCHMARK_BATCHES:
        break


elapsed = time.time() - start


# ==================================================
# Results
# ==================================================

time_per_batch = (
    elapsed
    / NUM_BENCHMARK_BATCHES
)

estimated_epoch_time = (
    time_per_batch
    * len(loader)
)


print()
print("=" * 50)
print("Benchmark Results")
print("=" * 50)

print(
    "Batches:",
    NUM_BENCHMARK_BATCHES
)

print(
    "Total time:",
    round(elapsed, 2),
    "seconds"
)

print(
    "Time/batch:",
    round(time_per_batch, 3),
    "seconds"
)

print(
    "Batches/epoch:",
    len(loader)
)

print(
    "Estimated epoch time:",
    round(
        estimated_epoch_time / 60,
        2
    ),
    "minutes"
)

print(
    "Estimated 10 epochs:",
    round(
        estimated_epoch_time * 10 / 60,
        2
    ),
    "minutes"
)