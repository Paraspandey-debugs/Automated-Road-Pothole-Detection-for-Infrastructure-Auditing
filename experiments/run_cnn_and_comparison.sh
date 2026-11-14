#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="/home/hyperghost/github/Automated-Road-Pothole-Detection-for-Infrastructure-Auditing"
cd "$REPO_DIR"

mkdir -p runs results/models results/tables results/figures

LOG_FILE="runs/cnn_baseline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================================"
echo "Starting Deep CNN Baseline & Classical RF Comparison Pipeline"
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Working directory: $(pwd)"
echo "Host: $(uname -n) | Kernel: $(uname -r)"
echo "Python: $(python3 --version)"
echo "PyTorch CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo "================================================================================"

# Verify Random Forest models are strictly preserved
if [ ! -f "results/models/e1_random_forest.joblib" ]; then
    echo "ERROR: Primary Random Forest model results/models/e1_random_forest.joblib missing!"
    exit 1
fi
if [ ! -f "results/models/e1_random_forest_backup.joblib" ]; then
    echo "Backing up Random Forest model to results/models/e1_random_forest_backup.joblib..."
    cp "results/models/e1_random_forest.joblib" "results/models/e1_random_forest_backup.joblib"
fi
echo "[OK] Random Forest model verified and backed up."
echo "     Primary size: $(du -h results/models/e1_random_forest.joblib | cut -f1)"
echo "     Backup size:  $(du -h results/models/e1_random_forest_backup.joblib | cut -f1)"

# Step 1: Train ResNet-18 Deep Baseline
echo ""
echo ">>> STEP 1/3: Fine-tuning ResNet-18 Deep Baseline (25 Epochs)..."
python3 -u experiments/e5_cnn_baseline.py \
    --epochs 25 \
    --batch-size 32 \
    --lr 1e-4 \
    --weight-decay 1e-3 \
    --workers 4

echo "[OK] Step 1 complete. ResNet-18 best checkpoint saved."

# Step 2: Head-to-Head Comparison with Random Forest
echo ""
echo ">>> STEP 2/3: Running Head-to-Head Benchmark (RF vs. CNN)..."
python3 -u experiments/compare_rf_vs_cnn.py

echo "[OK] Step 2 complete. Tables and comparison curves generated."

# Step 3: Grad-CAM Explainability Visualization
echo ""
echo ">>> STEP 3/3: Generating Grad-CAM Spatial Explainability Heatmaps..."
python3 -u experiments/gradcam_visualizer.py

echo "[OK] Step 3 complete. Grad-CAM visual diagnosis generated."

# Step 4: Sync figures to artifact brain directory
ARTIFACT_DIR="/home/hyperghost/.gemini/antigravity-cli/brain/eac3cfa9-5881-4e33-b9dc-b577118e905b"
if [ -d "$ARTIFACT_DIR" ]; then
    echo ""
    echo ">>> Syncing results to artifact storage..."
    cp results/figures/e5_cnn_training_curves.png "$ARTIFACT_DIR/" || true
    cp results/figures/rf_vs_cnn_pr_curves.png "$ARTIFACT_DIR/" || true
    cp results/figures/rf_vs_cnn_roc_curves.png "$ARTIFACT_DIR/" || true
    cp results/figures/rf_vs_cnn_confusion_matrices.png "$ARTIFACT_DIR/" || true
    cp results/figures/rf_vs_cnn_tradeoff_comparison.png "$ARTIFACT_DIR/" || true
    cp results/figures/cnn_gradcam_explanations.png "$ARTIFACT_DIR/" || true
    echo "[OK] Artifacts copied."
fi

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY AT $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================================"
