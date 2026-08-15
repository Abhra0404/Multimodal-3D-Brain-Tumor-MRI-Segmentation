# Brain Tumor MRI Segmentation

A deep learning project for **3D brain tumor segmentation from multimodal MRI scans** using the BraTS-GLI dataset.

## Overview

The project aims to segment brain tumor sub-regions from:

- T1-native (`t1n`)
- T1-contrast (`t1c`)
- T2-weighted (`t2w`)
- T2-FLAIR (`t2f`)

The goal is to build a complete pipeline:

```text
MRI Volumes
    ↓
Preprocessing
    ↓
3D Segmentation Model
    ↓
Tumor Mask
    ↓
Dice / IoU / HD95
    ↓
Visualization & Deployment
````

## Current Progress

* [x] Dataset setup
* [x] NIfTI loading with NiBabel
* [x] MRI visualization
* [x] Segmentation mask analysis
* [x] Tumor-region visualization
* [x] Tumor volume calculation
* [ ] Reusable patient loader
* [ ] 2D U-Net baseline
* [ ] 3D U-Net
* [ ] Multimodal experiments
* [ ] Advanced architectures
* [ ] Deployment

## Dataset

**ASNR-MICCAI-BraTS2023-GLI Challenge Training Dataset**

Each patient contains:

```text
t1n  → T1-native
t1c  → T1-contrast
t2w  → T2-weighted
t2f  → T2-FLAIR
seg  → Segmentation mask
```

Segmentation labels:

```text
0 → Background
1 → Necrotic / non-enhancing tumor
2 → Edema
3 → Enhancing tumor
```

## Tech Stack

* Python 3.11
* PyTorch
* MONAI
* NiBabel
* NumPy
* Matplotlib
* scikit-image

## Project Structure

```text
brain-tumor-mri-segmentation/
├── data/
├── notebooks/
├── src/
├── models/
├── results/
├── requirements.txt
├── pyproject.toml
└── README.md
```

> Dataset files and model checkpoints are excluded from Git.

## Evaluation

The project will primarily use:

* Dice Score
* IoU
* Precision
* Recall
* HD95

## Disclaimer

For **educational and research purposes only**. This project is not a clinical diagnostic tool.

## Author

**Abhra Jaiswal**

> Stay focused, stay productive, and keep leveling up! — kaizenX out. ✌️