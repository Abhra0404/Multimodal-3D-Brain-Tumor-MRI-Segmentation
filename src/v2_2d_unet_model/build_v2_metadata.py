from pathlib import Path
import json
import random


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
TRAIN_RATIO = 0.80


# --------------------------------------------------
# Find patients
# --------------------------------------------------

patient_ids = sorted(
    p.name
    for p in DATASET_DIR.iterdir()
    if p.is_dir()
)

print(f"Total cases: {len(patient_ids)}")


# --------------------------------------------------
# Group related cases
# --------------------------------------------------

groups = {}

for patient_id in patient_ids:

    # Remove the final case suffix.
    # Example:
    # BraTS-GLI-00008-000
    # BraTS-GLI-00008-001
    #
    # → BraTS-GLI-00008

    group_id = patient_id.rsplit("-", 1)[0]

    groups.setdefault(
        group_id,
        []
    ).append(patient_id)


group_ids = sorted(groups)

print(f"Unique patient groups: {len(group_ids)}")


# --------------------------------------------------
# Shuffle groups
# --------------------------------------------------

rng = random.Random(SEED)

rng.shuffle(group_ids)


# --------------------------------------------------
# Train / validation split
# --------------------------------------------------

split_index = int(
    len(group_ids) * TRAIN_RATIO
)

train_groups = sorted(
    group_ids[:split_index]
)

val_groups = sorted(
    group_ids[split_index:]
)


train_ids = sorted(
    patient_id
    for group_id in train_groups
    for patient_id in groups[group_id]
)

val_ids = sorted(
    patient_id
    for group_id in val_groups
    for patient_id in groups[group_id]
)


# --------------------------------------------------
# Save metadata
# --------------------------------------------------

metadata = {
    "seed": SEED,
    "train_ratio": TRAIN_RATIO,

    "total_cases": len(patient_ids),
    "total_groups": len(group_ids),

    "train_groups": train_groups,
    "val_groups": val_groups,

    "train_ids": train_ids,
    "val_ids": val_ids,
}


output_path = (
    OUTPUT_DIR
    / "patient_split.json"
)

with open(
    output_path,
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=2
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("Split complete")
print("----------------------------")
print("Total groups:", len(group_ids))
print("Training groups:", len(train_groups))
print("Validation groups:", len(val_groups))
print("Training cases:", len(train_ids))
print("Validation cases:", len(val_ids))
print()
print("Saved:", output_path)