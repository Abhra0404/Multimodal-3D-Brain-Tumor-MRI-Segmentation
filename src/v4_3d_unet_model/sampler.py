from pathlib import Path
import json

import numpy as np


class BalancedPatchSampler:

    def __init__(
        self,
        dataset,
        index_path,
        samples_per_epoch=4000,
        tumor_ratio=0.5,
        seed=42
    ):

        self.dataset = dataset

        self.samples_per_epoch = (
            samples_per_epoch
        )

        self.tumor_ratio = tumor_ratio

        self.rng = np.random.default_rng(
            seed
        )

        self.index_path = Path(
            index_path
        )

        self._load_index()

    # ==================================================
    # Load saved patch index
    # ==================================================

    def _load_index(self):

        if not self.index_path.exists():

            raise FileNotFoundError(
                f"Patch index not found:\n"
                f"{self.index_path}"
            )

        with open(
            self.index_path,
            "r"
        ) as f:

            index_data = json.load(f)

        self.tumor_indices = np.array(
            index_data["tumor_indices"],
            dtype=np.int64
        )

        self.background_indices = np.array(
            index_data["background_indices"],
            dtype=np.int64
        )

        print(
            "Loaded patch index:"
        )

        print(
            "Tumor patches:",
            len(self.tumor_indices)
        )

        print(
            "Background patches:",
            len(self.background_indices)
        )

    # ==================================================
    # Sample indices
    # ==================================================

    def sample_indices(self):

        tumor_count = int(
            self.samples_per_epoch
            * self.tumor_ratio
        )

        background_count = (
            self.samples_per_epoch
            - tumor_count
        )

        tumor_indices = self.rng.choice(
            self.tumor_indices,
            size=tumor_count,
            replace=True
        )

        background_indices = self.rng.choice(
            self.background_indices,
            size=background_count,
            replace=True
        )

        indices = np.concatenate(
            [
                tumor_indices,
                background_indices
            ]
        )

        self.rng.shuffle(
            indices
        )

        return indices.tolist()