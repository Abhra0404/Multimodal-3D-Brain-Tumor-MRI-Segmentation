# Multimodal 3D Brain Tumor MRI Segmentation

A deep learning project for automated brain tumor segmentation from multimodal MRI scans using the BraTS dataset.

The project explores how different deep learning approaches can segment tumor regions from multiple MRI modalities, with an emphasis on reproducible preprocessing, efficient training, quantitative evaluation, and qualitative error analysis.

---

## Overview

Brain tumor segmentation is a fundamental task in medical image analysis. Given multiple MRI modalities of a patient's brain, the goal is to identify and segment tumor regions accurately.

This project uses:

- **T1**
- **T1ce**
- **T2**
- **FLAIR**

as multimodal input to a deep learning segmentation pipeline.

The current pipeline processes MRI volumes into 2D axial slices and uses a U-Net-based architecture for tumor segmentation.

---

## Pipeline

```text
BraTS MRI Volumes
        ↓
Patient-Level Dataset Split
        ↓
Multimodal MRI Loading
        ↓
Per-Modality Normalization
        ↓
Tumor-Aware Slice Sampling
        ↓
128 × 128 Preprocessing
        ↓
PyTorch Dataset
        ↓
U-Net Segmentation Model
        ↓
BCE + Dice Loss
        ↓
Apple MPS Training
        ↓
Quantitative Evaluation
        ↓
Qualitative Error Analysis
````

---

## Model

The project currently uses a **2D U-Net** architecture.

### Input

```text
4 × 128 × 128
```

where the four channels correspond to:

```text
T1
T1ce
T2
FLAIR
```

### Output

```text
1 × 128 × 128
```

representing the predicted binary tumor mask.

---

## Dataset

The project uses the **BraTS (Brain Tumor Segmentation)** dataset.

Each patient contains multimodal MRI volumes along with a corresponding segmentation mask.

The dataset pipeline includes:

* Patient-level train/validation splitting
* Multimodal MRI loading
* Per-modality normalization
* Tumor-aware slice selection
* Balanced tumor/non-tumor sampling
* Efficient preprocessed storage

---

## Training

Current training configuration:

| Parameter     | Value         |
| ------------- | ------------- |
| Architecture  | 2D U-Net      |
| Input         | 4 × 128 × 128 |
| Batch Size    | 8             |
| Epochs        | 10            |
| Optimizer     | AdamW         |
| Learning Rate | 1e-4          |
| Loss          | BCE + Dice    |
| Hardware      | Apple M4 MPS  |

---

## Results

The current baseline was evaluated on **4,000 validation slices**.

| Metric           |      Score |
| ---------------- | ---------: |
| Tumor-Slice Dice | **0.8393** |
| IoU              | **0.8681** |
| Precision        | **0.9380** |
| Recall           | **0.9209** |
| Specificity      | **0.9992** |

The validation set contains:

* **1,992 tumor-containing slices**
* **2,008 empty slices**

Tumor-slice Dice is emphasized because it provides a more informative measure of segmentation quality on slices containing tumor regions.

---

## Qualitative Results

The project also evaluates predictions visually using:

```text
MRI
Ground Truth
Prediction
Overlay
```

Error analysis includes:

* Best-performing predictions
* Typical predictions
* Failure cases
* Small tumor regions
* False positives
* False negatives

![V2 Qualitative Results](results/v2/v2_qualitative_image.png)

---

## Project Structure

```text
BraTS/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── models/
│   └── v2/
│       ├── best.pt
│       └── latest.pt
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_v2_training.ipynb
│   └── 03_v2_evaluation.ipynb
│
├── src/
│   ├── dataset.py
│   ├── unet.py
│   ├── train_v2.py
│   └── benchmark_training.py
│
├── results/
│   └── v2/
|       └── v2_qualitative_image.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

* Python
* PyTorch
* NumPy
* NiBabel
* SciPy
* Matplotlib
* Jupyter Notebook

---

## Reproducibility

The project separates:

* Raw medical imaging data
* Preprocessed datasets
* Model checkpoints
* Training code
* Evaluation notebooks
* Experiment results

Patient-level splitting is used to prevent slices from the same patient appearing across training and validation sets.

---

## License

This project is intended for educational and research purposes.

The BraTS dataset is subject to its own data usage and licensing terms.


## Author

**Abhra Jaiswal**

> Stay focused, stay productive, and keep leveling up! — kaizenX out. ✌️
