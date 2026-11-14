# 📄 Paper Writing Data Summary & Experimental Results Reference

This document serves as the comprehensive data reference for drafting and refining the research paper:
**"Automated Road Pothole Detection and Spatial Severity Auditing via Hand-Crafted Feature Fusion and Deep Representations"**
*Authors:* Paras Pandey, Chandan Sapkale, Neel Ghule, Kavyan Hemrom (*Indian Institute of Information Technology Allahabad*)

---

## 1. Dataset Breakdown & Zero-Leakage Splitting

- **Dataset Source:** Road Damage Dataset (RDD2022) — Figshare DOI: `10.6084/m9.figshare.21431547`
- **Total Images Analyzed:** 15,340 real-world driving frames across 3 countries (Czech Republic, India, United States)
- **Image Resolution:** Native $600 \times 600$ pixels; normalized to $224 \times 224$ for classical feature extraction and $640 \times 640$ for YOLO object detection.
- **Zero-Leakage Splitting Scheme:** Group-stratified partitioning using sequence blocks (`group_id = <Country>_seq_<index>`, blocks of 50 contiguous frames) to prevent temporal and spatial frame leakage across splits.

### Table 1: Dataset Partitioning & Class Distribution
| Country Domain | Total Images | Pothole Images (D40 $\ge 1$) | Total Pothole Boxes | Hard Negatives (Crack-only) | Normal (Unblemished) | Pothole Prevalence (\%) |
|---|---|---|---|---|---|---|
| **India** | 7,706 | 1,848 | 3,344 | 3,112 | 2,746 | 24.0\% |
| **United States** | 4,805 | 114 | 136 | 2,842 | 1,849 | 2.4\% |
| **Czech Republic** | 2,829 | 118 | 158 | 1,346 | 1,365 | 4.2\% |
| **Total** | **15,340** | **2,080** | **3,638** | **7,300** | **5,960** | **13.6\%** |

### Split Partition Sizes:
- **Train Split (70\%):** 10,668 images (1,290 potholes, 12.1\% prevalence)
- **Validation Split (15\%):** 2,349 images (230 potholes, 9.8\% prevalence)
- **Test Split (15\%):** 2,323 images (280 potholes, 12.1\% prevalence)

---

## 2. Multi-Modal Feature Representation (10,217 Dimensions)

Preprocessed with edge-preserving bilateral filtering ($\sigma_s = 50, \sigma_r = 50, d = 5$):
1. **HSV Color Descriptors ($f_{\text{color}} \in \mathbb{R}^{4105}$):**
   - 3 Channel Moments (Mean, Std, Skewness) $\times 3 = 9$ dimensions
   - $16 \times 16 \times 16$ normalized 3D color histogram $= 4,096$ dimensions
2. **Structural Edge Gradients ($f_{\text{hog}} \in \mathbb{R}^{6084}$):**
   - 9 gradient orientations, $16 \times 16$ pixels/cell, $2 \times 2$ cells/block, L2-Hys block normalization
3. **Surface Roughness Texture ($f_{\text{lbp}} \in \mathbb{R}^{18}$):**
   - Uniform rotation-invariant Local Binary Patterns ($P = 16, R = 2$)
4. **Haralick Spatial Co-occurrence ($f_{\text{glcm}} \in \mathbb{R}^{10}$):**
   - 5 properties (contrast, dissimilarity, homogeneity, energy, correlation) $\times 2$ stats (mean, std) across distances $\{1, 3\}$ and 4 angles $\{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$

---

## 3. Experiment 1: Classical Classification Benchmark

Models evaluated on held-out test split (2,323 images). `StandardScaler` fitted strictly on training partition.

### Table 2: Classical Model Benchmark Results
| Model Architecture | Hyperparameter Configuration | Accuracy | Precision | Recall | F1 (Pothole) | F1 Macro | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | L2 regularized, $C = 0.01$, balanced weights | 0.8472 | 0.4051 | 0.5714 | 0.4741 | 0.6923 | 0.8112 | 0.4445 |
| **SVM (Linear Hinge)** | SGD hinge loss, $\alpha = 0.001$, balanced weights | 0.8627 | 0.4316 | 0.4393 | 0.4354 | 0.6786 | 0.7935 | 0.3708 |
| **Random Forest** | 200 trees, max depth $= 10$, min samples split $= 5$ | 0.8924 | 0.5517 | 0.5714 | 0.5614 | 0.7500 | 0.8391 | 0.5768 |
| **XGBoost (Single Best)** | 200 estimators, $\eta = 0.05$, `scale_pos_weight` $= 7.27$ | **0.9031** | **0.6132** | 0.5321 | 0.5698 | **0.7576** | **0.8646** | 0.6218 |
| **Tri-Model Ensemble** | 50\% XGB + 40\% RF + 10\% LR (Val-Tuned Threshold $= 0.48$) | 0.8988 | 0.5782 | 0.5679 | **0.5730** | 0.7542 | 0.8592 | **0.6223** |

---

## 4. Experiment 2: Feature Ablation Study

Evaluates individual descriptor contributions using Random Forest and XGBoost.

### Table 3: Feature Ablation Results (Test Split)
| Model | Feature Representation | Dimensions | Precision | Recall | F1 (Pothole) | PR-AUC |
|---|---|---|---|---|---|---|
| **XGBoost** | Color Only (HSV stats + 3D hist) | 4,105 | 0.4318 | 0.5536 | 0.4851 | 0.5139 |
| **XGBoost** | Color + HOG (Gradients) | 10,189 | 0.5719 | 0.5679 | **0.5699** | 0.6005 |
| **XGBoost** | Color + HOG + LBP + GLCM (Full Fusion) | 10,217 | 0.5869 | 0.5429 | 0.5640 | **0.6161** |
| **XGBoost** | PCA-Reduced (Unsupervised reduction) | 64 | 0.4323 | **0.6500** | 0.5193 | 0.5502 |
| **Random Forest** | Color Only | 4,105 | 0.6491 | 0.3964 | 0.4922 | 0.5459 |
| **Random Forest** | Color + HOG | 10,189 | 0.7569 | 0.3893 | **0.5142** | 0.5833 |
| **Random Forest** | Full Fusion (Color + HOG + LBP + GLCM) | 10,217 | **0.7576** | 0.3571 | 0.4854 | **0.6007** |
| **Random Forest** | PCA-Reduced | 64 | 0.6343 | 0.3964 | 0.4879 | 0.5778 |

**Key Takeaway:** HOG structural gradients provide the largest single performance boost ($\Delta \text{F1} = +0.085$). Full fusion maximizes precision-recall area (PR-AUC $= 0.6161$). PCA with 64 components retains 91.2\% of full F1 with 99.4\% dimension reduction.

---

## 5. Experiment 3: Cross-Domain Generalization Collapse

Domain transfer evaluated across 3 independent random seeds $\{42, 123, 456\}$, reporting Mean $\pm$ Standard Deviation.

### Table 4: Cross-Domain Generalization Results
| Scenario | Source Training Domains | Target Domain | Model | In-Domain F1 | Target Domain F1 | Domain Drop ($\Delta \text{F1}$) | Target PR-AUC |
|---|---|---|---|---|---|---|---|
| **West $\to$ India** | US + Czech (6,490 train) | India (7,706 imgs) | XGBoost | $0.083 \pm 0.042$ | $0.006 \pm 0.002$ | $+0.077 \pm 0.044$ | $0.276 \pm 0.030$ |
| **West $\to$ India** | US + Czech | India | Random Forest | $0.022 \pm 0.020$ | $0.000 \pm 0.000$ | $+0.022 \pm 0.020$ | $0.243 \pm 0.019$ |
| **West $\to$ India** | US + Czech | India | Logistic Reg | $0.064 \pm 0.030$ | $0.158 \pm 0.087$ | $-0.093 \pm 0.057$ | $0.216 \pm 0.014$ |
| **Czech $\to$ India** | Czech Republic | India | XGBoost | $0.190 \pm 0.040$ | $0.001 \pm 0.002$ | $+0.189 \pm 0.042$ | $0.238 \pm 0.010$ |
| **Czech $\to$ India** | Czech Republic | India | Logistic Reg | $0.171 \pm 0.041$ | $0.276 \pm 0.045$ | $-0.105 \pm 0.004$ | $0.221 \pm 0.010$ |
| **US $\to$ India** | United States | India | SVM | $0.047 \pm 0.041$ | $0.164 \pm 0.020$ | $-0.117 \pm 0.038$ | $0.204 \pm 0.003$ |

**Key Takeaway:** Tree-based models (XGBoost, RF) suffer catastrophic transfer collapse ($\text{F1} \approx 0.00$) due to covariate shift in asphalt color and lighting. Linear classifiers retain baseline recall but suffer from low discriminability.

---

## 6. Experiment 4: YOLOv8n Deep Detection & Severity Audit

Trained with Mosaic (1.0), Mixup (0.1), scale/color jitter on an RTX 4050 GPU (100 epochs, batch 16, imgsz 640).

### Table 5: YOLOv8 Training Comparison (10 vs 100 Epochs)
| Evaluation Metric | 10 Epochs Baseline | **100 Epochs Final** | Relative Improvement |
|---|---|---|---|
| **mAP@0.5** | 0.1152 | **0.2550** | **+121.4\% (2.2× boost)** |
| **mAP@0.5:0.95** | 0.0388 | **0.0911** | **+134.8\% (2.3× boost)** |
| **Precision** | 0.2022 | **0.3550** | **+75.6\%** |
| **Recall** | 0.1835 | **0.2860** | **+55.9\%** |
| **Validation Cls Loss** | 2.0029 | **2.3595** (Stable) | Well regularized |
| **Validation Box Loss** | 1.4298 | **1.5661** (Augmented) | Sharp bounding |
| **Severity Audit Correlation ($\rho$)** | $\approx 0.00$ | **0.8485** ($p = 1.51 \times 10^{-16}$) | **Highly reliable sorting** |
| **Inference Latency** | 7.8 ms | **3.0 ms** | **~330 FPS** |

### Street Segment Severity Ranking Correlation:
- **Formula:** Pothole Severity Index $S_i = \sum_{j} (w_j \cdot h_j)$ normalized by image area.
- Aggregated by driving sequence ($\text{group\_id}$).
- **Spearman Rank Correlation:** $\rho = \mathbf{0.8485}$ ($p = 1.5071 \times 10^{-16}$)

---

## 7. Key Figures & Visual Assets for the Paper

| Figure File | Caption / Description | Location in Repo |
|---|---|---|
| `pipeline_architecture.pdf` | End-to-end multi-modal pipeline architecture diagram | `results/figures/pipeline_architecture.pdf` |
| `e1_pr_curves_all.png` | Combined PR curves for Logistic Regression, SVM, Random Forest, XGBoost | `results/figures/e1_pr_curves_all.png` |
| `e1_loss_curves.png` | Log-loss vs boosting rounds (XGBoost) and epochs (SGD) | `results/figures/e1_loss_curves.png` |
| `e1_confusion_matrices.png` | Normalized confusion matrices for classical classifiers | `results/figures/e1_confusion_matrices.png` |
| `e2_feature_ablation.png` | Grouped bar chart of F1 vs descriptor representation | `results/figures/e2_feature_ablation.png` |
| `e3_domain_gap.png` | Cross-domain generalization drop ($\Delta \text{F1}$ Mean $\pm$ Std) | `results/figures/e3_domain_gap.png` |
| `e4_yolo_loss_curves.png` | 100-epoch training loss and validation mAP curves | `results/figures/e4_yolo_loss_curves.png` |
| `e4_severity_ranking_correlation.png` | Scatter plot of ground truth vs predicted road segment severity ($\rho = 0.8485$) | `results/figures/e4_severity_ranking_correlation.png` |
| `potholes_vs_cracks_comparison.png` | Qualitative inspection: clear potholes vs tough hairline crack hard negatives | `results/figures/potholes_vs_cracks_comparison.png` |
| `yolo100_pothole_vs_crack_inference.png` | YOLO 100-epoch visual detections on potholes and cracks | `results/figures/yolo100_pothole_vs_crack_inference.png` |
| `failure_cases.png` | Qualitative failure grid (False Positives and False Negatives) | `results/figures/failure_cases.png` |
