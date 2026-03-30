# FloodENet: A Lightweight ENet with Coordinate Attention for Real-Time Flood Segmentation

> **APSIPA ASC 2026** — School of Electrical and Electronic Engineering, Hanoi University of Science and Technology

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

**FloodENet** is a lightweight semantic segmentation model designed for real-time flood area detection. Built upon the ENet backbone, FloodENet incorporates:

- **Improved Coordinate Attention (Imp. CA)** — a deeper shared bottleneck better suited to the irregular spatial extents of floodwater regions
- **Depthwise Separable Convolutions (DWS)** — replacing standard 3×3 convolutions in Regular Bottlenecks to reduce model complexity (~9× MAC reduction)
- **Dual-Path Downsampling (AP)** — combining max-pooling and average-pooling in the shortcut branch for complementary feature aggregation

With only **0.25M parameters**, FloodENet achieves:

| Dataset | IoU | DSC | Params (M) | FPS |
|---|---|---|---|---|
| Flood Area Segmentation (FAS) | **0.779 ± 0.003** | **0.876 ± 0.002** | **0.25** | 68.32 |
| AIFloodSense (AIFSN) | **0.735 ± 0.007** | **0.848 ± 0.005** | **0.25** | 67.54 |

---

## Architecture

```
Encoder:
  InitialBlock          →  16 × 128 × 128
  Bottleneck 1.0 (down) →  64 × 64 × 64
  4× Bottleneck 1.x     →  64 × 64 × 64
  Bottleneck 2.0 (down) → 128 × 32 × 32
  Bottleneck 2.1–2.8    → 128 × 32 × 32   (regular / dilated / asymmetric, DWS)
  Repeat Section 2      → 128 × 32 × 32

Decoder:
  Imp. CA               → 128 × 32 × 32
  Bottleneck 4.0 (up)   →  64 × 64 × 64
  Bottleneck 4.1–4.2    →  64 × 64 × 64
  Imp. CA               →  64 × 64 × 64
  Bottleneck 5.0 (up)   →  16 × 128 × 128
  Bottleneck 5.1        →  16 × 128 × 128
  FullConv (transposed) →   C × 256 × 256
```

---

## Installation

```bash
git clone https://github.com/doantrongthai/Real_Time_Flood_Segmentation
cd Real_Time_Flood_Segmentation
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 2.0
- albumentations
- opencv-python
- gdown
- tqdm

---

## Datasets

### Flood Area Segmentation (FAS)
290 aerial RGB images with binary pixel-wise masks. Split: 70% train / 15% val / 15% test.

### AIFloodSense (AIFSN)
376 training images + 94 test images. Non-flood classes are consolidated into a single background category for binary segmentation.

Download datasets automatically:

```bash
python main.py --model floodeNet --download --dataset floodkaggle
python main.py --model floodeNet --download --dataset floodscene
```

---

## Training

**Single run:**
```bash
python main.py \
  --model floodeNet \
  --dataset floodkaggle \
  --loss bce_dice \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.001 \
  --size 256 \
  --seed 42
```

**Multi-seed experiment (recommended for paper results):**
```bash
python main.py \
  --model floodeNet \
  --dataset floodkaggle \
  --loss bce_dice \
  --epochs 50 \
  --batch_size 4 \
  --multiseed \
  --seeds 42 123 456 789 2024
```

**Reproducibility check:**
```bash
python main.py \
  --model floodeNet \
  --dataset floodkaggle \
  --loss bce_dice \
  --verify_repro
```

---

## Training Details

| Setting | Value |
|---|---|
| GPU | NVIDIA T4 (15 GB) |
| Epochs | 50 |
| Batch size | 4 |
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Initial LR | 1e-3 |
| LR schedule | Cosine Annealing → 1e-6 |
| Loss | 0.5 × BCE + 0.5 × Dice |
| Input size | 256 × 256 |
| Augmentation | HorizontalFlip, VerticalFlip, RandomBrightnessContrast, ShiftScaleRotate |
| Evaluation | 5 seeds: 42, 123, 456, 789, 2024 |

---

## Results

Results are reported as **mean ± std** over 5 independent runs.

### Comparison with baselines (FAS dataset)

| Model | IoU | DSC | Params (M) | FPS |
|---|---|---|---|---|
| SegNet | 0.713 ± 0.013 | 0.833 ± 0.009 | 29.48 | 50.26 |
| U-Net | 0.757 ± 0.006 | 0.862 ± 0.004 | 31.04 | 26.57 |
| ESPNetv2 | 0.760 ± 0.005 | 0.864 ± 0.003 | 0.84 | **126.39** |
| SegFormer | 0.766 ± 0.015 | 0.867 ± 0.010 | 24.72 | 35.67 |
| ENet | 0.773 ± 0.005 | 0.872 ± 0.003 | 0.37 | 80.84 |
| **FloodENet (Ours)** | **0.779 ± 0.003** | **0.876 ± 0.002** | **0.25** | 68.32 |

### Ablation study

| Base | AP | DWS | CA | Imp. CA | Params (M) | AIFSN IoU | FAS IoU |
|---|---|---|---|---|---|---|---|
| ✓ | | | | | 0.37 | 0.725 | 0.773 |
| ✓ | ✓ | | | | 0.37 | 0.728 | 0.771 |
| ✓ | | ✓ | | | 0.23 | 0.714 | 0.776 |
| ✓ | ✓ | ✓ | ✓ | | 0.23 | **0.735** | 0.777 |
| ✓ | ✓ | ✓ | | ✓ | 0.25 | **0.735** | **0.779** |

---

## Project Structure

```
Real_Time_Flood_Segmentation/
├── models/
│   ├── __init__.py
│   └── floodeNet.py          # FloodENet architecture
├── utils/
│   ├── trainer.py            # Training loop
│   ├── dataloader.py         # Dataset & DataLoader
│   └── metrics.py            # IoU, Dice, FPS, GFLOPs
├── losses/
│   ├── __init__.py
│   └── bce_dice.py           # BCE + Dice combined loss
├── main.py                   # Entry point
└── README.md
```

---

## Citation

```bibtex
@inproceedings{vu2026floodeNet,
  title     = {FloodENet: A Lightweight ENet with Coordinate Attention for Real-Time Flood Segmentation},
  author    = {Vu, Tuan Anh and Doan, Trong Thai and Dao, Duc Thinh and
               Nguyen, Thi Lan Huong and Nguyen, Tuan Ninh and Nguyen, Thi Hue and
               Nguyen, Hoang Nam and Nguyen, Duc Thuan and Hoang, Si Hong and
               Hoang, Manh Cuong},
  booktitle = {2026 Asia Pacific Signal and Information Processing Association
               Annual Summit and Conference (APSIPA ASC)},
  year      = {2026}
}
```

---

## Acknowledgment

This work was supported by the LIDP Laboratory, Hanoi University of Science and Technology.