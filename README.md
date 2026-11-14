# Automated Road Pothole Detection & Infrastructure Severity Auditing

An end-to-end computer vision and machine learning framework for automated road surface inspection and municipal maintenance auditing using the multi-national Road Damage Dataset (RDD2022).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLO-v8n-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Framework Overview

Municipal pavement auditing traditionally depends on manual visual surveys or expensive laser profilometers. This project establishes an automated, reproducible inspection pipeline combining **multi-modal hand-crafted feature fusion** (Color, HOG, LBP, GLCM) and **YOLOv8 deep representations** for segment-level road severity prioritization:

1. **Zero-Leakage Group Splitting:** Sequence-aware partitioning by 50-frame contiguous blocks to eliminate spatial/temporal leakage across train, validation, and test splits.
2. **10,217-Dimensional Multi-Modal Feature Extraction:** Fusing HSV color moments, 3D histograms, structural HOG gradients, LBP surface roughness, and Haralick GLCM co-occurrence texture.
3. **Classical ML Benchmark (E1):** Evaluating Logistic Regression, SVM, Random Forest, XGBoost, and a soft-voting ensemble under group-stratified cross-validation.
4. **Feature Ablations & Cross-Domain Transfer (E2 & E3):** Quantifying descriptor marginal gains and measuring cross-domain generalization collapse across national road networks (Czech Republic, India, USA).
5. **Spatial Severity Auditing (E4):** Fine-tuning YOLOv8n to compute cavity area ratios and establishing rank correlation ($\rho = 0.8485$) with ground-truth segment deterioration.

---

## 📊 Summary of Experimental Results

### Classical Model Benchmark (Test Split: 2,323 images)
| Model | Accuracy | Precision | Recall | F1 Pothole | F1 Macro | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 84.72% | 0.4051 | 0.5714 | 0.4741 | 0.6923 | 0.8112 | 0.4445 |
| **SVM (Linear Hinge)** | 86.27% | 0.4316 | 0.4393 | 0.4354 | 0.6786 | 0.7935 | 0.3708 |
| **Random Forest** | 89.24% | 0.5517 | 0.5714 | 0.5614 | 0.7500 | 0.8391 | 0.5768 |
| **XGBoost (Best Single)** | **90.31%** | **0.6132** | 0.5321 | 0.5698 | **0.7576** | **0.8646** | 0.6218 |
| **Tri-Model Ensemble** | 89.88% | 0.5782 | 0.5679 | **0.5730** | 0.7542 | 0.8592 | **0.6223** |

### YOLOv8n Detection & Infrastructure Severity Audit (100 Epochs)
- **mAP@0.5:** **0.2550** (Up from 0.1152 in baseline, **+121.4% gain**)
- **mAP@0.5:0.95:** **0.0911** (Up from 0.0388, **+134.8% gain**)
- **Inference Latency:** **3.0 ms / frame** on RTX 4050 GPU (~330 FPS)
- **Infrastructure Audit Spearman Rank Correlation:** **$\rho = 0.8485$** ($p = 1.51 \times 10^{-16}$)

---

## 📂 Repository Structure

```
├── configs/                  # Hyperparameter YAML configuration files
│   ├── base.yaml             # Paths, preprocessing, hardware settings
│   ├── data.yaml             # YOLO dataset configuration
│   ├── experiments.yaml      # Experiment parameters (E1, E2, E3, E4)
│   └── features.yaml         # Descriptor dimensions & extraction parameters
├── experiments/              # Executable experiment benchmark scripts
│   ├── e1_models.py          # Classical classifier benchmark
│   ├── e2_feats.py           # Multi-modal feature ablation study
│   ├── e3_cross.py           # Cross-domain generalization benchmark
│   ├── e4_audit.py           # YOLO detection & street severity audit
│   ├── failure_cases.py      # False positive & false negative analysis
│   └── visualize_potholes_vs_cracks.py # Potholes vs crack comparison
├── paper/                    # IEEE conference paper source and assets
│   ├── main.tex              # Complete IEEEtran LaTeX manuscript
│   ├── main.pdf              # Compiled 6-page paper PDF
│   ├── references.bib        # BibTeX bibliography
│   └── PAPER_DATA_SUMMARY.md # Master data & metrics reference for paper writing
├── results/                  # Generated figures, tables, and serialized models
│   ├── figures/              # Publication figures (PR curves, loss curves, audits)
│   ├── models/               # Trained .joblib classical classifiers
│   └── tables/               # Exported CSV result tables
├── scripts/
│   └── draw_pipeline.py      # Architecture flow diagram generator
└── src/
    ├── data/                 # Data download, label parsing, splits, VOC->YOLO
    └── feats/                # Color, HOG, LBP, and GLCM extractors
```

---

## 🚀 Quickstart & Pipeline Execution

```bash
# 1. Clone repository and install dependencies
git clone https://github.com/hyperghost/Automated-Road-Pothole-Detection-for-Infrastructure-Auditing.git
cd Automated-Road-Pothole-Detection-for-Infrastructure-Auditing
pip install -r requirements.txt

# 2. Run complete end-to-end pipeline via Makefile:
make all

# Or run individual stages:
make data      # Download datasets
make yolo      # VOC XML to YOLO format
make labels    # Parse metadata and labels.csv
make splits    # Group-aware zero-leakage splits
make feats     # Extract 10,217-dim feature vectors
make e1        # Run classical model benchmark
make e2        # Run feature ablation study
make e3        # Run cross-domain benchmark
make e4        # Run YOLO detection & severity audit
```

---

## 📜 Citation

```bibtex
@inproceedings{pandey2026pothole,
  title={Automated Road Pothole Detection and Spatial Severity Auditing via Hand-Crafted Feature Fusion and Deep Representations},
  author={Pandey, Paras and Sapkale, Chandan and Ghule, Neel and Hemrom, Kavyan},
  booktitle={IEEE Conference Proceedings},
  year={2026}
}
```
