from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from v2_2d_unet_model.dataset import BraTSSliceDataset
from v2_2d_unet_model.unet import UNet


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

dataset = BraTSSliceDataset(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v2"
    / "train"
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

model = UNet().to(device)

images, masks = next(iter(loader))

images = images.to(device)
masks = masks.to(device)

print("Device:", device)
print("Images:", images.shape)
print("Masks:", masks.shape)

with torch.no_grad():

    logits = model(images)

print("Logits:", logits.shape)