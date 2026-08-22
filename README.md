# Multimodal 3D Brain Tumor MRI Segmentation

A deep learning pipeline for brain tumor segmentation from multimodal MRI scans using the **BraTS dataset**.

The project progressively explores 2D and 2.5D segmentation approaches while maintaining patient-level train/validation separation and quantitative evaluation.

---

## Overview

Brain tumor segmentation is the task of identifying tumor regions from MRI scans.

This project uses four MRI modalities:

- T1
- T1ce
- T2
- FLAIR

The pipeline covers:

- MRI preprocessing
- Slice selection
- Patient-level dataset splitting
- Multimodal image construction
- U-Net segmentation
- 2.5D adjacent-slice context
- Model training and checkpointing
- Quantitative evaluation
- Failure-mode analysis
- Prediction visualization

---

## Dataset

The project uses the **BraTS 2023 Glioma Segmentation dataset**.

Each patient contains four MRI modalities and a segmentation mask.

```text
T1
T1ce
T2
FLAIR
  ↓
Segmentation Mask
````

Patient-level splitting is used to prevent slices from the same patient appearing in both training and validation sets.

---

## Project Versions

### V1 — Baseline Segmentation

Established the initial preprocessing and segmentation pipeline.

* Multimodal MRI input
* 2D U-Net segmentation
* Basic preprocessing
* Dataset validation
* Initial training and evaluation pipeline

---

### V2 — 2D U-Net

V2 established the main 2D segmentation baseline.

#### Input

```text
4 × 128 × 128
```

The four channels correspond to:

```text
T1
T1ce
T2
FLAIR
```

#### Model

```text
2D U-Net
```

#### Training

```text
Loss: BCE + Dice
Optimizer: AdamW
Learning Rate: 1e-4
Epochs: 10
Batch Size: 8
Device: Apple MPS
```

#### Results

| Metric               |         V2 |
| -------------------- | ---------: |
| Best Validation Dice | **0.8240** |
| Validation IoU       | **0.7919** |
| Overall Pixel Dice   | **0.9294** |
| Tumor Slice Dice     | **0.8393** |
| Precision            | **0.9380** |
| Recall               | **0.9209** |
| Specificity          | **0.9992** |

---

### V3 — 2.5D Adjacent-Slice U-Net

V3 introduces neighboring-slice context while keeping the underlying U-Net architecture unchanged.

Instead of using only slice `z`, three consecutive slices are combined:

```text
z-1
 z
z+1
```

Each slice contains four MRI modalities:

```text
       z-1        z        z+1
    ┌────────┐ ┌────────┐ ┌────────┐
    │ T1     │ │ T1     │ │ T1     │
    │ T1ce   │ │ T1ce   │ │ T1ce   │
    │ T2     │ │ T2     │ │ T2     │
    │ FLAIR  │ │ FLAIR  │ │ FLAIR  │
    └────────┘ └────────┘ └────────┘

             ↓

        12-channel input
          128 × 128
```

#### Input

```text
12 × 128 × 128
```

#### Model

```text
2D U-Net
```

The network architecture remains the same as V2; only the input representation changes from 4 to 12 channels.

#### Training

```text
Loss: BCE + Dice
Optimizer: AdamW
Learning Rate: 1e-4
Epochs: 10
Batch Size: 8
Device: Apple MPS
```

#### Results

Best checkpoint:

```text
Epoch: 9
```

| Metric                   |         V3 |
| ------------------------ | ---------: |
| **Best Validation Dice** | **0.8399** |
| Validation IoU           | **0.8079** |
| Overall Pixel Dice       | **0.9277** |
| Tumor Slice Dice         | **0.8508** |
| Precision                | **0.9177** |
| Recall                   | **0.9380** |
| Specificity              | **0.9989** |

### V2 → V3

| Metric           |     V2 |         V3 |
| ---------------- | -----: | ---------: |
| Mean Slice Dice  | 0.8240 | **0.8399** |
| Tumor Slice Dice | 0.8393 | **0.8508** |
| IoU              | 0.7919 | **0.8079** |
| Recall           | 0.9209 | **0.9380** |
| Precision        | 0.9380 |     0.9177 |

The V3 results show that adding adjacent-slice context improved slice-level segmentation and tumor recall.

---

## V3 Failure Analysis

The full validation set contains:

```text
Validation slices: 4000
Tumor-containing slices: 1992
Empty slices: 2008
```

### Tumor Size Performance

| Tumor Size              | Slices |  Mean Dice |
| ----------------------- | -----: | ---------: |
| Small (<100 pixels)     |    409 |     0.5758 |
| Medium (100–499 pixels) |    858 |     0.8998 |
| Large (≥500 pixels)     |    725 | **0.9479** |

Small tumors remain the most challenging cases.

For V3:

```text
Small tumor slices: 409
Dice < 0.25: 107
Dice < 0.50: 138
Dice ≥ 0.75: 201
Dice = 0: 0
```

Qualitative analysis shows that the remaining failures are primarily associated with very small lesions where accurate localization is difficult.

---

## Evaluation

The evaluation pipeline measures:

### Segmentation Metrics

* Dice coefficient
* IoU
* Precision
* Recall
* Specificity

### Slice-Level Analysis

* Mean slice Dice
* Tumor-containing slice Dice
* Empty-slice performance
* Small/medium/large tumor performance
* Best and worst performing slices

### Qualitative Analysis

Prediction visualizations are used to inspect:

* Correct segmentations
* False positives
* False negatives
* Small tumor failures
* Boundary errors

---

## Project Structure

```text
BraTS/
│
├── data/
│   ├── raw/
│   ├── metadata/
│   └── processed/
│       ├── v2/
│       └── v3/
│
├── models/
│   ├── v2/
│   │   ├── best.pt
│   │   └── latest.pt
│   │
│   └── v3/
│       ├── best.pt
│       └── latest.pt
│
├── notebooks/
│   └── ...
│
├── src/
│   ├── v2_2d_unet_model/
│   │   ├── dataset.py
│   │   ├── preprocess_v2.py
│   │   ├── train_v2.py
│   │   └── unet.py
│   │
│   └── v3_2.5d_unet_model/
│       ├── dataset.py
│       ├── preprocess_v3.py
│       ├── train_v3.py
│       └── unet_v3.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tech Stack

* Python
* PyTorch
* NumPy
* NiBabel
* Matplotlib
* Jupyter
* Apple Metal Performance Shaders (MPS)

---

## Reproducibility

The project uses:

* Patient-level train/validation splitting
* Fixed preprocessing pipelines
* Saved dataset metadata
* Saved model checkpoints
* Recorded training metrics
* Separate versioned preprocessing and model implementations

Best-performing checkpoints are stored under:

```text
models/v2/best.pt
models/v3/best.pt
```

---

## Key Results

The progression demonstrates the effect of adding spatial context:

```text
V2 — 2D single slice
        ↓
     Dice: 0.8240

        ↓

V3 — 2.5D adjacent slices
        ↓
     Dice: 0.8399
```

The V3 experiment improved the validation Dice by:

```text
+0.0159
```

while also increasing tumor-slice recall from:

```text
0.9209 → 0.9380
```

---

## License

This project is intended for educational and research purposes.

## Author

**Abhra Jaiswal**

> Stay focused, stay productive, and keep leveling up! — kaizenX out. ✌️
