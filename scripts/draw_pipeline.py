#!/usr/bin/env python3
"""
draw_pipeline.py – Renders a high-resolution vector pipeline diagram (PDF & PNG)
for the IEEEtran conference paper, illustrating the end-to-end framework.
"""

from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt

out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 6.5)
ax.axis("off")

# Helper to draw rounded boxes
def draw_box(ax, x, y, w, h, text, title=None, facecolor="#e1f5fe", edgecolor="#0288d1", lw=1.5, text_color="#01579b", fontsize=9, title_size=10):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12,rounding_size=0.15",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(rect)
    if title:
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=text_color, zorder=3)
        ax.text(x + w / 2, y + (h - 0.35) / 2, text, ha="center", va="center", fontsize=fontsize, color="#263238", zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=text_color, zorder=3)
    return rect

# Helper to draw arrows
def draw_arrow(ax, x1, y1, x2, y2, color="#455a64", lw=1.8, style="-|>", dashed=False):
    ls = "--" if dashed else "-"
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            linestyle=ls,
            shrinkA=2,
            shrinkB=2,
            mutation_scale=14,
        ),
        zorder=4,
    )

# 1. Dataset Box
draw_box(
    ax, 0.4, 3.8, 2.0, 2.2,
    "• Czech (2,829 imgs)\n• India (7,706 imgs)\n• US (4,805 imgs)\nTotal: 15,340 images\nPotholes: 1,800 imgs\nHard Negatives: 7,300",
    title="Road Image Data",
    facecolor="#ede7f6", edgecolor="#512da8", text_color="#311b92", fontsize=8.5
)

# 2. Preprocessing & Group Splitting
draw_box(
    ax, 2.9, 3.8, 2.3, 2.2,
    "• Group-Aware Split\n  (Zero-leakage sequences)\n• Bilateral Denoising\n• Resize to 224×224\n• Train / Val / Test\n  (70 / 15 / 15)",
    title="Preprocessing",
    facecolor="#e8f5e9", edgecolor="#2e7d32", text_color="#1b5e20", fontsize=8.5
)

# 3. Feature Extraction
draw_box(
    ax, 5.7, 3.8, 2.5, 2.2,
    "• Color/HSV (4,105 dims)\n  (3D Hist + Channel Stats)\n• HOG (6,084 dims)\n• LBP (18 dims)\n• Haralick / GLCM (10 dims)\nTotal: 10,217 dims\n• StandardScaler (Train only)",
    title="Feature Extraction",
    facecolor="#fff3e0", edgecolor="#e65100", text_color="#bf360c", fontsize=8.5
)

# 4. Classical Classifiers & Tuning
draw_box(
    ax, 8.7, 3.8, 2.4, 2.2,
    "• Logistic Regression\n• SVM (RBF Kernel)\n• Random Forest\n• XGBoost Classifier\n• 3-Fold GroupKFold CV\n  Hyperparameter Tuning",
    title="Classical ML Models",
    facecolor="#e0f7fa", edgecolor="#00838f", text_color="#006064", fontsize=8.5
)

# 5. Core Evaluation & Cross-Domain
draw_box(
    ax, 11.6, 3.8, 2.0, 2.2,
    "• Accuracy, Recall, F1\n• Precision-Recall AUC\n• Loss-vs-Epoch Curves\n• Cross-Domain Delta:\n  (US+Czech -> India)\n• Qualitative Errors",
    title="Model Evaluation",
    facecolor="#fce4ec", edgecolor="#c2185b", text_color="#880e4f", fontsize=8.5
)

# Arrows for main pipeline
draw_arrow(ax, 2.4, 4.9, 2.9, 4.9)
draw_arrow(ax, 5.2, 4.9, 5.7, 4.9)
draw_arrow(ax, 8.2, 4.9, 8.7, 4.9)
draw_arrow(ax, 11.1, 4.9, 11.6, 4.9)

# Extension / Branch: Object Detection & Infrastructure Audit (Lower half)
# Section boundary line
ax.plot([0.4, 13.6], [3.2, 3.2], color="#b0bec5", linestyle=":", lw=1.2)
ax.text(7.0, 3.35, "Optional Severity Auditing Extension (YOLOv8 Detection)", ha="center", va="center", fontsize=10, fontweight="bold", color="#546e7a")

# Branch arrow from Raw Images down to YOLOv8
draw_arrow(ax, 1.4, 3.8, 1.4, 2.1, color="#78909c", dashed=True)
draw_arrow(ax, 1.4, 2.1, 2.9, 2.1, color="#78909c", dashed=True)

draw_box(
    ax, 2.9, 0.9, 2.5, 2.0,
    "• YOLOv8n Fine-tuning\n• Input: 640×640 images\n• Single-class D40 cavity\n• Pretrained backbone\n• Device: RTX 4050 GPU",
    title="YOLOv8n Detector",
    facecolor="#efebe9", edgecolor="#4e342e", text_color="#3e2723", fontsize=8.5
)

draw_box(
    ax, 6.0, 0.9, 2.8, 2.0,
    "• Predicted Pothole Boxes\n• Normalized Box Area\n  Area = sum(w * h)\n• Frame Severity Index\n  (Cavity Area / Road Area)",
    title="Spatial Cavity Metric",
    facecolor="#fff8e1", edgecolor="#f57f17", text_color="#f57f17", fontsize=8.5
)

draw_box(
    ax, 9.4, 0.9, 3.5, 2.0,
    "• Street Segment Aggregation (group_id)\n• Severity Ranking (Low / Med / High)\n• Spearman Rank Correlation (rho)\n  Ground Truth vs. Predicted Area Ratio\n• Automated Infrastructure Audit Report",
    title="Road Infrastructure Audit",
    facecolor="#e8eaf6", edgecolor="#283593", text_color="#1a237e", fontsize=8.5
)

draw_arrow(ax, 5.4, 1.9, 6.0, 1.9, color="#78909c", dashed=True)
draw_arrow(ax, 8.8, 1.9, 9.4, 1.9, color="#78909c", dashed=True)

plt.tight_layout()
pdf_path = out_dir / "pipeline_architecture.pdf"
png_path = out_dir / "pipeline_architecture.png"
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved pipeline architecture to {pdf_path} and {png_path}")
