from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSSliceDataset(Dataset):

    def __init__(self, data_dir):

        self.data_dir = Path(data_dir)

        shard_paths = sorted(
            self.data_dir.glob("shard_*.npz")
        )

        if not shard_paths:
            raise RuntimeError(
                f"No shards found in {self.data_dir}"
            )

        images = []
        masks = []

        print(
            f"Loading {len(shard_paths)} shards "
            f"from {self.data_dir}"
        )

        for path in shard_paths:

            print(f"  Loading {path.name}")

            with np.load(path) as data:

                images.append(
                    data["images"]
                )

                masks.append(
                    data["masks"]
                )

        self.images = np.concatenate(
            images,
            axis=0
        )

        self.masks = np.concatenate(
            masks,
            axis=0
        )

        print(
            f"Loaded {len(self.images)} samples"
        )

        print(
            f"Images memory: "
            f"{self.images.nbytes / 1024**3:.2f} GB"
        )

        print(
            f"Masks memory: "
            f"{self.masks.nbytes / 1024**3:.2f} GB"
        )

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image = torch.from_numpy(
            self.images[idx]
        )

        mask = torch.from_numpy(
            self.masks[idx]
        )

        return image, mask