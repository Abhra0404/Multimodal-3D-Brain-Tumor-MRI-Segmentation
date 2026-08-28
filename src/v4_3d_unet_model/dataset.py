from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTS3DDataset(Dataset):

    def __init__(self, data_dir):

        self.data_dir = Path(data_dir)

        self.files = sorted(
            self.data_dir.glob("*.npz")
        )

        if not self.files:

            raise RuntimeError(
                f"No .npz patches found in {self.data_dir}"
            )

        print(
            f"Found {len(self.files)} 3D patches"
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        data = np.load(
            self.files[index]
        )

        image = data["image"]
        mask = data["mask"]

        image = torch.from_numpy(
            image
        ).float()

        mask = torch.from_numpy(
            mask
        ).float()

        return image, mask