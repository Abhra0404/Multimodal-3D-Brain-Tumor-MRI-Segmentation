import torch

from v2_2d_unet_model.unet import UNet


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = UNet().to(device)

x = torch.randn(
    2,
    4,
    128,
    128,
    device=device
)

with torch.no_grad():

    y = model(x)

print("Device:", device)
print("Input:", x.shape)
print("Output:", y.shape)