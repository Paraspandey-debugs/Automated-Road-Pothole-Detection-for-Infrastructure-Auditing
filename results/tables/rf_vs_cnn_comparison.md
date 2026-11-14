# Head-to-Head Comparison: Random Forest vs. ResNet-18

Evaluated on identical held-out test split (N = 2323 images, 280 potholes, 2,043 normal/crack roads).

| Model                     | Input Pipeline                             | Parameters / Size   |   Feature Extr. Latency (ms) |   Inference Latency (ms) |   Total Edge Latency (ms) |   Throughput (FPS) |   Accuracy |   Precision |   Recall |   F1 (Pothole) |   F1 (Macro) |   ROC-AUC |   PR-AUC |
|:--------------------------|:-------------------------------------------|:--------------------|-----------------------------:|-------------------------:|--------------------------:|-------------------:|-----------:|------------:|---------:|---------------:|-------------:|----------:|---------:|
| Random Forest (Baseline)  | 10,217-dim Hand-Crafted (HOG+LBP+GLCM+HSV) | 7.00 MB             |                         38.2 |                    23.01 |                     61.21 |               16.3 |     0.8924 |      0.5517 |   0.5714 |         0.5614 |       0.75   |    0.8391 |   0.5768 |
| ResNet-18 (Deep Baseline) | Raw 224x224 RGB Images (End-to-End)        | 11.17M (128.04 MB)  |                          0   |                     1.95 |                      1.95 |              513.4 |     0.9169 |      0.6548 |   0.6571 |         0.656  |       0.8044 |    0.9038 |   0.7264 |

### Latency Breakdown Notes:
- **Random Forest:** Requires ~38.2 ms to extract 10,217 hand-crafted features per image + 23.01 ms tree inference = **61.21 ms total** (16.3 FPS on CPU).
- **ResNet-18:** Direct RGB pixel ingestion (0 ms feature extraction). GPU inference: **1.95 ms** (513.4 FPS on RTX 4050 Laptop GPU); CPU inference: **19.43 ms** (51.5 FPS).
