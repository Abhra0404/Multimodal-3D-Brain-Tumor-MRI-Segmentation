from v2_2d_unet_model.dataset import BraTSSliceDataset


train_dataset = BraTSSliceDataset(
    "data/processed/v2/train"
)

val_dataset = BraTSSliceDataset(
    "data/processed/v2/val"
)

print(
    "Train samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


x, y = train_dataset[0]

print()
print("Input shape:", x.shape)
print("Target shape:", y.shape)

print(
    "Input dtype:",
    x.dtype
)

print(
    "Target dtype:",
    y.dtype
)

print(
    "Target values:",
    y.unique()
)