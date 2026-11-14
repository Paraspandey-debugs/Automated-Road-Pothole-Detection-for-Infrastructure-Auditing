# 📊 Comprehensive Experimental Run Observations: Deep ResNet-18 vs. Hand-Crafted Random Forest

**Project:** Automated Road Pothole Detection for Infrastructure Auditing  
**Authors:** Paras Pandey (IIT2024235), Chandan Sapkale (IIT2024258), Neel Ghule (IIT2024266), Kavyan Hemrom (IIT2024245)  
**Affiliation:** Indian Institute of Information Technology Allahabad  
**Execution Timestamp:** 2026-11-14 18:33 UTC  
**Evaluation Dataset:** Road Damage Dataset 2022 (RDD2022), Group-Stratified Zero-Leakage Sequence Split  

---

## 1. Hardware, Environment & Dataset Partitions

### Hardware & Runtime Environment:
- **Operating System:** Linux (Fedora 44, Kernel `7.1.8-200.fc44.x86_64`)
- **GPU Accelerator:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6 VRAM)
- **CPU:** Intel Multi-Core Host with 12 parallel threads
- **PyTorch Version:** `2.14.0+cu130`
- **CUDA Device:** `cuda:0`

### Dataset Partitions (Group-Stratified Zero-Leakage Split):
*Partitioned by 50-frame continuous capture sequence blocks (`group_id = <Country>_seq_<index>`) to strictly prevent temporal and spatial frame overlap between training, validation, and test splits.*

| Partition | Total Images | Pothole Positives ($y = 1$) | Normal & Crack Negatives ($y = 0$) | Pothole Class Prevalence | Loss Balancing Weight ($w_{\text{pos}}$) |
|---|---|---|---|---|---|
| **Training Split** | **10,668** | 1,290 | 9,378 | 12.09% | **7.27** ($9378 / 1290$) |
| **Validation Split** | **2,349** | 230 | 2,119 | 9.79% | N/A (unweighted evaluation) |
| **Test Split (Held-Out)** | **2,323** | 280 | 2,043 | 12.05% | N/A (unweighted evaluation) |
| **Total Benchmark** | **15,340** | 1,800 | 13,540 | 11.73% | — |

---

## 2. Deep CNN Architecture & Training Setup

### ResNet-18 Configuration:
- **Base Backbone:** Pretrained ResNet-18 (ImageNet-1K weights)
- **Head Adaptation:** Replaced 1,000-class linear projection with `nn.Sequential(nn.Dropout(p=0.3), nn.Linear(512, 1))`
- **Total Parameters:** 11,177,025 (11.18 Million)
- **Input Dimensions:** Raw $224 \times 224 \times 3$ RGB pixel tensors (Normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`)
- **Data Augmentation (Train only):** Random horizontal flip ($p = 0.5$), random rotation ($\pm 10^\circ$), color jitter (brightness, contrast, saturation $\pm 0.2$)
- **Loss Function:** `nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7.27]))`
- **Optimizer:** AdamW ($\text{learning rate} = 10^{-4}$, $\text{weight decay} = 10^{-3}$)
- **Learning Rate Scheduler:** `CosineAnnealingLR` ($T_{\max} = 25$, $\eta_{\min} = 10^{-6}$)
- **Batch Size:** 32 (Train), 64 (Val & Test)
- **Epochs:** 25 full passes (1,052.5 seconds total runtime = 17.54 minutes, ~42.1 seconds per epoch)
- **Checkpointing Strategy:** Saved checkpoint whenever validation PR-AUC achieved a new maximum.

---

## 3. Full 25-Epoch Training Progression (Exact Numeric Records)

*All metrics recorded during execution and saved to `results/tables/e5_cnn_training_metrics.csv`:*

| Epoch | Train Loss | Val Loss | Val PR-AUC | Val F1 (Pothole) | Val Precision | Val Recall | Val Accuracy | Learning Rate | Status / Event |
|---|---|---|---|---|---|---|---|---|---|
| **01** | 0.9209 | 0.7529 | 0.5933 | 0.4762 | 0.3620 | 0.6957 | 0.8872 | $9.96 \times 10^{-5}$ | ★ New Best Checkpoint |
| **02** | 0.7567 | 0.7666 | 0.6197 | 0.5774 | 0.5154 | 0.6565 | 0.9161 | $9.84 \times 10^{-5}$ | ★ New Best Checkpoint |
| **03** | 0.6743 | 0.6851 | 0.6619 | 0.5175 | 0.3814 | 0.8043 | 0.8944 | $9.65 \times 10^{-5}$ | ★ New Best Checkpoint |
| **04** | 0.5870 | 0.6729 | 0.6471 | 0.4972 | 0.3644 | 0.7826 | 0.8897 | $9.38 \times 10^{-5}$ | |
| **05** | 0.5054 | 0.7353 | 0.6400 | 0.4858 | 0.3523 | 0.7826 | 0.8838 | $9.05 \times 10^{-5}$ | |
| **06** | 0.4329 | 0.7837 | 0.6035 | 0.5319 | 0.4140 | 0.7435 | 0.9029 | $8.64 \times 10^{-5}$ | |
| **07** | 0.3648 | 0.8371 | 0.6716 | 0.5617 | 0.4482 | 0.7522 | 0.9119 | $8.18 \times 10^{-5}$ | ★ New Best Checkpoint |
| **08** | 0.3368 | 1.0482 | 0.6457 | 0.5953 | 0.5865 | 0.6043 | 0.9268 | $7.65 \times 10^{-5}$ | |
| **09** | 0.2839 | 0.9152 | 0.6270 | 0.5476 | 0.4497 | 0.7000 | 0.9115 | $7.09 \times 10^{-5}$ | |
| **10** | 0.2377 | 1.0416 | 0.6535 | 0.5342 | 0.4271 | 0.7130 | 0.9063 | $6.47 \times 10^{-5}$ | |
| **11** | 0.2127 | 1.1539 | 0.6562 | 0.5934 | 0.5329 | 0.6696 | 0.9229 | $5.83 \times 10^{-5}$ | |
| **12** | 0.1833 | 1.3356 | 0.6474 | 0.5949 | 0.5779 | 0.6130 | 0.9276 | $5.17 \times 10^{-5}$ | |
| **13** | 0.1557 | 1.1765 | 0.6432 | 0.5840 | 0.5204 | 0.6652 | 0.9208 | $4.50 \times 10^{-5}$ | |
| **14** | 0.1261 | 1.2343 | 0.6452 | 0.5869 | 0.5278 | 0.6609 | 0.9221 | $3.83 \times 10^{-5}$ | |
| **15** | 0.1085 | 1.3305 | 0.6716 | 0.5858 | 0.5131 | 0.6826 | 0.9204 | $3.18 \times 10^{-5}$ | |
| **16** | 0.0807 | 1.5190 | 0.6684 | 0.6157 | 0.6184 | 0.6130 | 0.9327 | $2.56 \times 10^{-5}$ | |
| **17** | 0.0654 | 1.5284 | 0.6715 | 0.6300 | 0.6384 | 0.6217 | 0.9357 | $1.98 \times 10^{-5}$ | |
| **18** | 0.0613 | 1.4914 | 0.6666 | 0.6125 | 0.5880 | 0.6391 | 0.9315 | $1.46 \times 10^{-5}$ | |
| **19** | 0.0625 | 1.6144 | 0.6543 | 0.6253 | 0.6634 | 0.5913 | 0.9366 | $1.00 \times 10^{-5}$ | |
| **20** | 0.0397 | 1.6745 | 0.6696 | 0.6330 | 0.6699 | 0.6000 | 0.9378 | $6.23 \times 10^{-6}$ | |
| **21** | 0.0368 | 1.6887 | 0.6769 | 0.6316 | 0.6667 | 0.6000 | 0.9374 | $3.27 \times 10^{-6}$ | ★ New Best Checkpoint |
| **22** | 0.0389 | 1.5989 | **0.6836** | **0.6421** | 0.6407 | **0.6435** | 0.9370 | $1.20 \times 10^{-6}$ | ★ **Global Best Checkpoint** |
| **23** | 0.0314 | 1.6370 | 0.6818 | 0.6430 | 0.6561 | 0.6304 | 0.9378 | $1.00 \times 10^{-6}$ | |
| **24** | 0.0267 | 1.7519 | 0.6831 | 0.6345 | 0.6732 | 0.6000 | 0.9383 | $1.00 \times 10^{-6}$ | |
| **25** | 0.0256 | 1.6655 | 0.6827 | 0.6327 | 0.6441 | 0.6217 | 0.9366 | $1.00 \times 10^{-6}$ | Final Epoch |

**Selected Optimal Checkpoint:** Epoch 22  
- Validation PR-AUC: **0.6836**  
- Validation F1: **0.6421** (Precision: 0.6407, Recall: 0.6435)  
- Checkpoint Path: `results/models/e5_resnet18_best.pt`

---

## 4. Head-to-Head Comparative Benchmark on Test Split ($N = 2,323$)

*Both models evaluated on the identical held-out test split (280 true potholes, 2,043 negative road images).*

### Table: Complete Empirical Metric Comparison
| Performance Metric | Hand-Crafted Random Forest | Deep ResNet-18 (Deep Baseline) | Absolute Delta ($\Delta$) | Relative Gain (\%) |
|---|---|---|---|---|
| **Accuracy** | 0.8924 (89.24%) | **0.9169 (91.69%)** | +0.0245 (+2.45%) | +2.75% |
| **Precision** | 0.5517 (55.17%) | **0.6548 (65.48%)** | +0.1031 (+10.31%) | **+18.69%** |
| **Recall (Pothole Detection Rate)** | 0.5714 (57.14%) | **0.6571 (65.71%)** | +0.0857 (+8.57%) | **+15.00%** |
| **F1-Score (Pothole Class)** | 0.5614 | **0.6560** | +0.0946 | **+16.85%** |
| **F1-Score (Macro Average)** | 0.7500 | **0.8044** | +0.0544 | +7.25% |
| **ROC-AUC** | 0.8391 | **0.9038** | +0.0647 | +7.71% |
| **PR-AUC (Primary Imbalance Metric)**| 0.5768 | **0.7264** | +0.1496 | **+25.94%** |

### Raw Confusion Matrix Decomposition (Held-Out Test Set, $N = 2,323$):
- **Ground Truth Totals:** Positive ($y = 1$) = 280, Negative ($y = 0$) = 2,043.
- **Random Forest Confusion Matrix:**
  - True Positives (TP): **160**
  - False Negatives (FN, Missed Potholes): **120**
  - False Positives (FP, False Alarms): **130**
  - True Negatives (TN): **1,913**
- **ResNet-18 CNN Confusion Matrix:**
  - True Positives (TP): **184** (+24 additional potholes detected)
  - False Negatives (FN, Missed Potholes): **96** (Missed pothole rate reduced from 42.86% to 34.29%)
  - False Positives (FP, False Alarms): **97** (False alarms reduced by 25.38%)
  - True Negatives (TN): **1,946**

---

## 5. Latency, Throughput & Hardware Footprint Benchmark

*Benchmarked on the target deployment machine (RTX 4050 Laptop GPU / Intel Multi-Core CPU).*

| Operational Metric | Hand-Crafted Random Forest | Deep ResNet-18 (Deep Baseline) | Advantage / Winner |
|---|---|---|---|
| **Input Representation** | 10,217-dim Hand-Crafted Vectors (HOG+LBP+GLCM+HSV) | Raw $224 \times 224$ RGB Tensor | ResNet-18 (End-to-End) |
| **Feature Extraction Overhead** | 38.20 ms / frame | **0.00 ms / frame** | ResNet-18 (**Zero Overhead**) |
| **Model Size on Disk** | **7.00 MB** (`.joblib`) | 44.7 MB weights / 128.0 MB checkpoint | Random Forest (**6.4× smaller**) |
| **Parameter Count** | N/A (200 Decision Trees) | 11,177,025 parameters | Random Forest |
| **Pure Inference Time (GPU)** | N/A (CPU execution only) | **1.95 ms / frame** (RTX 4050) | ResNet-18 (**513.4 FPS**) |
| **Pure Inference Time (CPU)** | 23.01 ms / frame | 19.43 ms / frame | ResNet-18 (Comparable CPU) |
| **Total Edge Latency (GPU Pipeline)**| 61.21 ms / frame (16.3 FPS) | **1.95 ms / frame (513.4 FPS)** | ResNet-18 (**31.4× faster**) |
| **Total Edge Latency (CPU Pipeline)**| 61.21 ms / frame (16.3 FPS) | **19.43 ms / frame (51.5 FPS)** | ResNet-18 (**3.1× faster**) |

### Practical Operational Takeaway:
1. **At Highway Inspection Speeds ($> 80$ km/h):** A patrol vehicle traveling at 80 km/h covers 22.2 meters per second. A camera capturing at 30 FPS produces a frame every 33.3 ms.
   - The Random Forest pipeline requires **61.21 ms** per frame, meaning it can only process **16.3 frames per second**, dropping roughly 46% of frames or failing real-time inspection.
   - ResNet-18 requires only **1.95 ms** per frame on an edge GPU, effortlessly processing **513.4 frames per second** with $> 95\%$ spare GPU capacity for simultaneous tracking, GPS tagging, and telemetry logging.
2. **At Ultra-Low-Cost Edge (Microcontrollers / No GPU):** Random Forest requires only 7.00 MB of storage and does not require CUDA runtime libraries or TensorRT engines, making it a viable fallback for microcontrollers with strict storage budgets ($< 16$ MB).

---

## 6. Spatial Explainability: Gini Importance vs. Grad-CAM

### Random Forest (Global Tabular Feature Attribution):
- **Feature Contribution:**
  - HOG structural edge gradient bins account for **64.2%** of the cumulative Gini impurity decrease across all 200 trees.
  - HSV color statistics and 3D histogram bins account for **24.5%**.
  - LBP and GLCM texture features account for **11.3%**.
- **Inherent Limitation:** Gini importance cannot generate a 2D spatial heatmap for an individual inspection image. It operates on global spatial cell aggregates, meaning a sharp linear road crack produces similar gradient histogram values to the boundary of a pothole, causing high false alarm rates (130 FP on test set).

### ResNet-18 (Local Spatial Saliency via Grad-CAM):
- **Target Layer:** `model.layer4` (the final residual convolutional block, output shape $7 \times 7 \times 512$).
- **True Pothole Positives (`India_000173`, `India_000209`, `India_000274`):**
  - Grad-CAM heatmaps generate intense, focused saliency ($> 0.85$ peak attention) precisely centered on the interior asphalt cavity and depression shadow.
  - Output pothole probability: $P(\text{pothole}) > 0.85$.
- **Severe Crack Hard Negatives (`Czech_000006`, `Czech_000020`, `Czech_000031`):**
  - Unlike HOG, Grad-CAM attention across longitudinal, transverse, and alligator cracks is either completely suppressed or diffused across the frame ($P(\text{pothole}) < 0.20$).
  - **Why CNN succeeds where hand-crafted fails:** Deep convolutional layers learn 2D depth perception, shadow continuity, and circular/elliptical depression aspect ratios, distinguishing hollow volume loss from superficial planar asphalt fractures.

---

## 7. Generated Visual Assets and Locations

All visual and tabular assets have been saved directly to the repository and are ready for citation:

1. **Training Loss & PR-AUC Curves:** `results/figures/e5_cnn_training_curves.png`
2. **Precision-Recall Curves Comparison:** `results/figures/rf_vs_cnn_pr_curves.png`
3. **ROC Curves Comparison:** `results/figures/rf_vs_cnn_roc_curves.png`
4. **Side-by-Side Confusion Matrices:** `results/figures/rf_vs_cnn_confusion_matrices.png`
5. **Multi-Dimensional Trade-off Bar Chart:** `results/figures/rf_vs_cnn_tradeoff_comparison.png`
6. **Grad-CAM Visual Explanations (6-sample grid):** `results/figures/cnn_gradcam_explanations.png`
7. **Numeric Comparison Table:** `results/tables/rf_vs_cnn_comparison.csv`
8. **ResNet-18 Best Checkpoint:** `results/models/e5_resnet18_best.pt`
9. **Random Forest Preserved Model:** `results/models/e1_random_forest.joblib` & `results/models/e1_random_forest_backup.joblib`
