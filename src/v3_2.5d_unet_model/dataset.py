from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTS25DDataset(Dataset):

    def __init__(self, data_dir):

        self.data_dir = Path(data_dir)

        self.files = sorted(
            self.data_dir.glob("*.npz")
        )

        if not self.files:
            raise RuntimeError(
                f"No .npz files found in {self.data_dir}"
            )

        # --------------------------------------------------
        # Build index:
        # (patient_file, slice_index)
        # --------------------------------------------------

        self.index = []

        for file_path in self.files:

            with np.load(
                file_path,
                mmap_mode="r"
            ) as data:

                num_slices = (
                    data["images"].shape[0]
                )

            for slice_idx in range(
                num_slices
            ):

                self.index.append(
                    (
                        file_path,
                        slice_idx
                    )
                )

        print(
            f"Loaded {len(self.files)} patients "
            f"with {len(self.index)} samples"
        )

    def __len__(self):

        return len(self.index)

    def __getitem__(
        self,
        idx
    ):

        file_path, slice_idx = (
            self.index[idx]
        )

        data = np.load(
            file_path
        )

        image = data[
            "images"
        ][slice_idx]

        mask = data[
            "masks"
        ][slice_idx]

        image = torch.from_numpy(
            image.copy()
        ).float()

        mask = torch.from_numpy(
            mask.copy()
        ).float()

        return image, mask