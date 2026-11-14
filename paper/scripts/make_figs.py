"""Regenerate paper figures directly from the repository result tables."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from PIL import Image

R = Path(sys.argv[1])          # repo results dir
OUT = Path(sys.argv[2])        # output figs dir
T = R / "tables"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C = {"LR": "#4C72B0", "SVM": "#55A868", "RF": "#DD8452", "XGBoost": "#C44E52",
     "India": "#C44E52", "Czech": "#4C72B0", "United_States": "#55A868"}
FULLW, COLW = 7.16, 3.5

# ---------------------------------------------------------------- deep curves
cnn = pd.read_csv(T / "e5_cnn_training_metrics.csv")
yolo = pd.read_csv(T / "e4_yolo_training_metrics.csv")
fig, ax = plt.subplots(1, 4, figsize=(FULLW, 1.6))
a = ax[0]
a.plot(cnn.epoch, cnn.train_loss, color="#4C72B0", label="Train")
a.plot(cnn.epoch, cnn.val_loss, color="#C44E52", ls="--", label="Validation")
a.axvline(22, color="0.4", ls=":", lw=0.9)
a.set(xlabel="Epoch", ylabel="Weighted BCE loss", title="(a) ResNet-18 loss", xlim=(1, 25))
a.legend(frameon=False, loc="upper left")
a = ax[1]
a.plot(cnn.epoch, cnn.val_pr_auc, color="#8172B2", marker="o", ms=2, label="Val PR-AUC")
a.plot(cnn.epoch, cnn.val_f1_pothole, color="#937860", marker="s", ms=2, label="Val F1 (pothole)")
a.axhline(0.098, color="0.5", ls="--", lw=0.8)
a.text(2, 0.12, "chance PR-AUC = 0.098", ha="left", fontsize=6.3, color="0.35")
a.axvline(22, color="0.4", ls=":", lw=0.9)
a.text(21.5, 0.80, "selected (ep. 22)", ha="right", va="top", fontsize=6.3, color="0.3")
a.set(xlabel="Epoch", ylabel="Score", title="(b) ResNet-18 validation", ylim=(0, 0.85), xlim=(1, 25))
a.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.38))
a = ax[2]
tr = yolo["train/box_loss"] + yolo["train/cls_loss"] + yolo["train/dfl_loss"]
va = yolo["val/box_loss"] + yolo["val/cls_loss"] + yolo["val/dfl_loss"]
ok = va > 1.0   # epoch 3 validation record is corrupted (sum 0.33) -> omitted
a.plot(yolo.epoch, tr, color="#4C72B0", label="Train")
a.plot(yolo.epoch[ok], va[ok], color="#C44E52", ls="--", label="Validation")
a.set_yscale("log")
a.set_yticks([4, 6, 10, 20]); a.set_yticklabels(["4", "6", "10", "20"])
a.minorticks_off()
a.set(xlabel="Epoch", ylabel="Box + cls + DFL loss (log)", title="(c) YOLOv8n loss", xlim=(1, 100))
a.legend(frameon=False, loc="upper right")
a = ax[3]
a.plot(yolo.epoch, yolo["metrics/mAP50(B)"], color="#C44E52", label="mAP50")
a.plot(yolo.epoch, yolo["metrics/mAP50-95(B)"], color="#8172B2", label="mAP50-95")
a.plot(yolo.epoch, yolo["metrics/precision(B)"], color="#4C72B0", lw=0.8, alpha=0.7, label="Precision")
a.plot(yolo.epoch, yolo["metrics/recall(B)"], color="#55A868", lw=0.8, alpha=0.7, label="Recall")
a.set(xlabel="Epoch", ylabel="Validation score", title="(d) YOLOv8n validation", xlim=(1, 100), ylim=(0, 0.75))
a.legend(frameon=False, loc="upper left", ncol=2)
for a in ax:
    a.grid(alpha=0.25, lw=0.4)
fig.tight_layout(w_pad=0.8)
fig.savefig(OUT / "fig_deep_curves.pdf")
plt.close(fig)

# ---------------------------------------------------------------- lift
# Country-level prevalence (pothole images / images), consistent with 1,800 total positives.
prev = {"India": 1530 / 7706, "Czech": 154 / 2829, "US": 116 / 4805}
prev["US+Czech"] = (154 + 116) / (2829 + 4805)
src = {"source_to_india": "US+Czech", "us_to_india": "US", "czech_to_india": "Czech"}
raw = pd.read_csv(T / "e3_cross_domain_raw_runs.csv")
raw["src"] = raw.scenario.map(src)
raw["lift_in"] = raw.in_pr_auc / raw.src.map(prev)
raw["lift_out"] = raw.out_pr_auc / prev["India"]
raw["model"] = raw.model.replace({"Logistic_Regression": "LR", "Random_Forest": "RF"})
g = raw.groupby(["scenario", "model"]).agg(li=("lift_in", "mean"), lis=("lift_in", "std"),
                                            lo=("lift_out", "mean"), los=("lift_out", "std")).reset_index()
g.to_csv(OUT / "lift_table.csv", index=False)
models = ["LR", "SVM", "RF", "XGBoost"]
order = ["source_to_india", "czech_to_india", "us_to_india"]
titles = {"source_to_india": "(a) US + Czech $\\rightarrow$ India",
          "czech_to_india": "(b) Czech $\\rightarrow$ India",
          "us_to_india": "(c) US $\\rightarrow$ India"}
fig, ax = plt.subplots(1, 3, figsize=(FULLW, 1.6), sharey=True)
x = np.arange(len(models)); w = 0.38
for a, sc in zip(ax, order):
    d = g[g.scenario == sc].set_index("model").loc[models]
    a.bar(x - w / 2, d.li, w, yerr=d.lis, color=[C[m] for m in models], alpha=0.45,
          edgecolor=[C[m] for m in models], lw=0.6, capsize=1.5, error_kw={"lw": 0.6})
    a.bar(x + w / 2, d.lo, w, yerr=d.los, color=[C[m] for m in models], hatch="////",
          edgecolor="white", lw=0, capsize=1.5, error_kw={"lw": 0.6})
    for i, m in enumerate(models):
        a.text(x[i] - w / 2, d.li[m] + d.lis[m] + 0.08, f"{d.li[m]:.1f}", ha="center", fontsize=5.8)
        a.text(x[i] + w / 2, d.lo[m] + d.los[m] + 0.08, f"{d.lo[m]:.1f}", ha="center", fontsize=5.8)
    a.axhline(1, color="k", ls="--", lw=0.8)
    a.set_xticks(x); a.set_xticklabels(models)
    a.set_title(titles[sc]); a.grid(axis="y", alpha=0.25, lw=0.4)
ax[0].set_ylabel("Lift = PR-AUC / prevalence")
ax[0].set_ylim(0, 6.3)
handles = [plt.Rectangle((0, 0), 1, 1, fc="0.75", ec="0.4", lw=0.6),
           plt.Rectangle((0, 0), 1, 1, fc="0.45", hatch="////", ec="white", lw=0),
           Line2D([], [], color="k", ls="--", lw=0.8)]
fig.legend(handles, ["Source-country test split (in-domain)", "India (unseen country)", "Chance level (lift = 1)"],
           frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(w_pad=0.6)
fig.savefig(OUT / "fig_lift.pdf")
plt.close(fig)

# ---------------------------------------------------------------- audit
d = pd.read_csv(T / "e4_segment_audit_ranking.csv")
d["c"] = d.group_id.str.split("_seq").str[0]
names = {"India": "India", "Czech": "Czech Rep.", "United_States": "United States"}
fig, ax = plt.subplots(1, 2, figsize=(COLW * 2, 1.7), gridspec_kw={"width_ratios": [1.25, 1]})
a = ax[0]
for c in ["Czech", "United_States", "India"]:
    s = d[d.c == c]
    a.scatter(100 * s.gt_segment_severity, 100 * s.pred_segment_severity, s=13, color=C[c],
              edgecolor="k", lw=0.3, alpha=0.85, label=f"{names[c]} ($n$={len(s)})", zorder=3)
m = 100 * d[["gt_segment_severity", "pred_segment_severity"]].values.max() * 1.05
a.plot([0, m], [0, m], color="0.4", ls="--", lw=0.8, label="$y = x$")
a.set(xlim=(-0.05, m), ylim=(-0.05, m), xlabel="Ground-truth segment severity (% of image area)",
      ylabel="Predicted segment severity (%)", title="(a) Predicted vs. ground-truth severity")
a.legend(frameon=False, loc="upper left"); a.grid(alpha=0.25, lw=0.4)
a = ax[1]
rows = [("Pooled", d)] + [(names[c], d[d.c == c]) for c in ["India", "Czech", "United_States"]]
cols = ["0.45", C["India"], C["Czech"], C["United_States"]]
vals = []
for (lab, s), col in zip(rows, cols):
    r, p = spearmanr(s.gt_segment_severity, s.pred_segment_severity)
    vals.append((lab, r, p, len(s)))
y = np.arange(len(vals))[::-1]
a.barh(y, [v[1] for v in vals], color=cols, height=0.6)
for yi, (lab, r, p, n) in zip(y, vals):
    ptxt = "$p<10^{-15}$" if p < 1e-15 else f"$p$={p:.2g}"
    a.text(r + 0.02, yi, f"$\\rho$={r:.2f}, {ptxt}, $n$={n}", va="center", fontsize=6.3)
a.set_yticks(y); a.set_yticklabels([v[0] for v in vals])
a.set(xlim=(0, 1.45), xlabel="Spearman $\\rho$", title="(b) Ranking agreement")
a.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); a.grid(axis="x", alpha=0.25, lw=0.4)
fig.tight_layout(w_pad=1.0)
fig.savefig(OUT / "fig_audit.pdf")
plt.close(fig)
print("audit", vals)

# ---------------------------------------------------------------- Grad-CAM (crop of repo figure)
im = Image.open(R / "figures" / "cnn_gradcam_explanations.png").convert("RGB")
rows_y = {1: (262, 1218), 2: (1376, 2332), 4: (3605, 4561), 6: (5834, 6790)}
orig_x, cam_x = (343, 1298), (1736, 2692)
info = [(1, "India, pothole", "$P$(pothole) = 0.998"), (2, "India, pothole", "$P$(pothole) = 1.000"),
        (4, "Czech, cracks", "$P$(pothole) = 0.000"), (6, "Czech, cracks", "$P$(pothole) = 0.004")]
fig, ax = plt.subplots(2, 4, figsize=(COLW, 2.0))
for j, (r, lab, p) in enumerate(info):
    y0, y1 = rows_y[r]
    ax[0, j].imshow(im.crop((orig_x[0], y0, orig_x[1], y1)))
    ax[1, j].imshow(im.crop((cam_x[0], y0, cam_x[1], y1)))
    ax[0, j].set_title(f"({'abcd'[j]}) {lab}", fontsize=6.3, pad=2)
    ax[1, j].set_xlabel(p.replace("$P$(pothole) = ", "$P$(pothole) = "), fontsize=6.0, labelpad=2)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout(h_pad=0.2, w_pad=0.15)
fig.savefig(OUT / "fig_gradcam.jpg", dpi=300, pil_kwargs={"quality": 90})
plt.close(fig)

# ---------------------------------------------------------------- crop title off E1 PR curve
pr = Image.open(OUT / "e1_pr_curves_all.png")
pass
lc = Image.open(OUT / "e1_loss_curves.png")
print(pr.size, lc.size)

# ---------------------------------------------------------------- YOLOv8n bounding boxes (crop of repo figure)
im = Image.open(R / "figures" / "yolo100_pothole_vs_crack_inference.png").convert("RGB")
rows_y = [(242, 1532), (1666, 2955)]
cols_x = [(120, 1410), (1605, 2895), (3090, 4380)]
panels = [(0, 0, "India: 2 boxes", "conf. 0.20, 0.23"),
          (0, 1, "India: 4 boxes", "conf. 0.21--0.24"),
          (0, 2, "India: 1 box", "conf. 0.23"),
          (1, 0, "Czech: 0 boxes", "cracks only"),
          (1, 1, "Czech: 0 boxes", "cracks only"),
          (1, 2, "Czech: 0 boxes", "cracks only")]
fig, ax = plt.subplots(1, 6, figsize=(FULLW, 1.55))
for k, (r, c, t, sub) in enumerate(panels):
    y0, y1 = rows_y[r]; x0, x1 = cols_x[c]
    ax[k].imshow(im.crop((x0, y0, x1, y1)))
    ax[k].set_title(f"({'abcdef'[k]}) {t}", fontsize=6.6, pad=2)
    ax[k].set_xlabel(sub.replace("--", "–"), fontsize=6.2, labelpad=2)
    ax[k].set_xticks([]); ax[k].set_yticks([])
fig.tight_layout(w_pad=0.25)
fig.savefig(OUT / "fig_yolo_boxes.jpg", dpi=300, pil_kwargs={"quality": 90})
plt.close(fig)

# ---------------------------------------------------------------- combined qualitative figure: YOLO boxes + Grad-CAM on the same six images
yim = Image.open(R / "figures" / "yolo100_pothole_vs_crack_inference.png").convert("RGB")
gim = Image.open(R / "figures" / "cnn_gradcam_explanations.png").convert("RGB")
y_rows = [(242, 1532), (1666, 2955)]; y_cols = [(120, 1410), (1605, 2895), (3090, 4380)]
g_rows = [(262, 1218), (1376, 2332), (2491, 3447), (3605, 4561), (4720, 5676), (5834, 6790)]
g_cam = (1736, 2692)
items = [  # (yolo row, yolo col, gradcam row, title, yolo note, cam prob)
    (0, 0, 0, "India_000173, pothole", "2 boxes (0.20, 0.23)", "0.998"),
    (0, 1, 1, "India_000209, pothole", "4 boxes (0.21–0.24)", "1.000"),
    (0, 2, 2, "India_000274, pothole", "1 box (0.23)", "0.992"),
    (1, 0, 3, "Czech_000006, cracks", "0 boxes", "0.000"),
    (1, 1, 4, "Czech_000020, cracks", "0 boxes", "0.002"),
    (1, 2, 5, "Czech_000031, cracks", "0 boxes", "0.004"),
]
fig, ax = plt.subplots(2, 6, figsize=(FULLW, 2.7))
for k, (yr, yc, gr, t, note, p) in enumerate(items):
    a = ax[0, k]; a.imshow(yim.crop((y_cols[yc][0], y_rows[yr][0], y_cols[yc][1], y_rows[yr][1])))
    a.set_title(f"({'abcdef'[k]}) {t.split(', ')[0]}, {t.split(', ')[1]}\nYOLOv8n: {note}", fontsize=6.2, pad=2)
    b = ax[1, k]; b.imshow(gim.crop((g_cam[0], g_rows[gr][0], g_cam[1], g_rows[gr][1])))
    b.set_xlabel(f"Grad-CAM: $P$(pothole) = {p}", fontsize=6.0, labelpad=1.5)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
ax[0, 0].set_ylabel("Detector boxes", fontsize=7)
ax[1, 0].set_ylabel("ResNet-18 Grad-CAM", fontsize=7)
fig.tight_layout(h_pad=0.3, w_pad=0.2)
fig.savefig(OUT / "fig_qualitative.jpg", dpi=300, pil_kwargs={"quality": 90})
plt.close(fig)
