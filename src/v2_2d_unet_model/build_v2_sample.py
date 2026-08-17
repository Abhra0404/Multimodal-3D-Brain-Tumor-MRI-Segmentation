from pathlib import Path
import json
import random
from collections import defaultdict


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "v2"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42

TRAIN_POSITIVE = 8_000
TRAIN_NEGATIVE = 8_000

VAL_POSITIVE = 2_000
VAL_NEGATIVE = 2_000


# --------------------------------------------------
# Load index
# --------------------------------------------------

with open(
    METADATA_DIR / "slice_index.json"
) as f:
    index = json.load(f)


# --------------------------------------------------
# Patient-balanced sampling
# --------------------------------------------------

def sample_by_patient(
    samples,
    n_samples,
    seed
):

    rng = random.Random(seed)

    grouped = defaultdict(list)

    for patient_id, slice_idx in samples:
        grouped[patient_id].append(
            slice_idx
        )

    patient_ids = list(grouped)

    rng.shuffle(patient_ids)

    for patient_id in patient_ids:
        rng.shuffle(
            grouped[patient_id]
        )

    selected = []

    while len(selected) < n_samples:

        added = False

        for patient_id in patient_ids:

            if grouped[patient_id]:

                slice_idx = (
                    grouped[patient_id].pop()
                )

                selected.append(
                    [
                        patient_id,
                        slice_idx
                    ]
                )

                added = True

                if len(selected) >= n_samples:
                    break

        if not added:
            break

    return selected


# --------------------------------------------------
# Sample
# --------------------------------------------------

train_positive = sample_by_patient(
    index["train"]["positive"],
    TRAIN_POSITIVE,
    SEED
)

train_negative = sample_by_patient(
    index["train"]["negative"],
    TRAIN_NEGATIVE,
    SEED + 1
)

val_positive = sample_by_patient(
    index["val"]["positive"],
    VAL_POSITIVE,
    SEED
)

val_negative = sample_by_patient(
    index["val"]["negative"],
    VAL_NEGATIVE,
    SEED + 1
)


# --------------------------------------------------
# Combine + shuffle
# --------------------------------------------------

train_samples = (
    train_positive
    + train_negative
)

val_samples = (
    val_positive
    + val_negative
)

random.Random(SEED).shuffle(
    train_samples
)

random.Random(SEED).shuffle(
    val_samples
)


# --------------------------------------------------
# Save
# --------------------------------------------------

sampled = {
    "seed": SEED,

    "train": train_samples,
    "val": val_samples,

    "counts": {
        "train": len(train_samples),
        "val": len(val_samples),
    }
}


output_path = (
    METADATA_DIR
    / "sampled_slices.json"
)

with open(
    output_path,
    "w"
) as f:
    json.dump(
        sampled,
        f
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("=" * 50)
print("V2 sampling complete")
print("=" * 50)

print(
    "Train positive:",
    len(train_positive)
)

print(
    "Train negative:",
    len(train_negative)
)

print(
    "Train total:",
    len(train_samples)
)

print()

print(
    "Validation positive:",
    len(val_positive)
)

print(
    "Validation negative:",
    len(val_negative)
)

print(
    "Validation total:",
    len(val_samples)
)

print()
print("Saved:", output_path)