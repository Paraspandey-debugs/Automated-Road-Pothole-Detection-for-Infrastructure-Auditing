#!/usr/bin/env python3
"""
gradcam_visualizer.py – Visual Explainability via Grad-CAM for Pothole Detection.

Generates Gradient-weighted Class Activation Mapping (Grad-CAM) visualizations from
ResNet-18 (layer4) to diagnose spatial attention and explainability:
  1. True Pothole Positives (India_000173, India_000209, India_000274)
  2. Tough Crack Hard Negatives (Czech_000006, Czech_000020, Czech_000031)

Demonstrates how deep feature representations discriminate circular depressions
from linear asphalt distress without relying on rigid hand-crafted heuristics.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class GradCAM:
    """Computes Grad-CAM heatmap for a target PyTorch model layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Runs forward & backward pass and computes 2D normalized Grad-CAM heatmap."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        logit = output[0, 0]
        prob = torch.sigmoid(logit).item()

        # Backward pass w.r.t pothole logit
        logit.backward()

        # Global average pooling of gradients
        # activations: [1, C, H, W], gradients: [1, C, H, W]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to keep only features that positively influence pothole classification
        cam = torch.clamp(cam, min=0)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, prob


def build_resnet18_binary() -> nn.Module:
    """Builds ResNet-18 architecture with binary classification head."""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, 1),
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM Explainability Visualizations.")
    default_model = (
        Path("results/models/e5_resnet18_weights.pt")
        if Path("results/models/e5_resnet18_weights.pt").exists()
        else Path("results/models/e5_resnet18_best.pt")
    )
    parser.add_argument("--model-path", type=Path, default=default_model)
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/features/feature_manifest.csv"))
    parser.add_argument("--out-path", type=Path, default=Path("results/figures/cnn_gradcam_explanations.png"))
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    log.info("Grad-CAM running on device: %s", device)

    # Load model
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    model = build_resnet18_binary().to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    log.info("Model loaded successfully from %s", args.model_path)

    # Attach Grad-CAM to layer4 (last convolutional residual block)
    grad_cam = GradCAM(model=model, target_layer=model.layer4)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    manifest_df = pd.read_csv(args.manifest)

    # Key diagnostic samples to explain
    sample_queries = [
        # (Identifier, Category Label, Diagnostic Description)
        ("India_000173", "True Pothole Positive", "Prominent deep circular pothole with sharp asphalt cavity"),
        ("India_000209", "True Pothole Positive", "Clustered irregular potholes on worn bituminous surface"),
        ("India_000274", "True Pothole Positive", "Large severe pothole directly in vehicle wheel path"),
        ("Czech_000006", "Crack Hard Negative", "Longitudinal / transverse surface fatigue crack without depression"),
        ("Czech_000020", "Crack Hard Negative", "Interconnected alligator crack pattern (high texture contrast)"),
        ("Czech_000031", "Crack Hard Negative", "Linear reflection road seam and superficial surface fracture"),
    ]

    selected_samples = []
    for query_id, category, desc in sample_queries:
        match = manifest_df[manifest_df["image_path"].str.contains(query_id)]
        if not match.empty:
            img_path = match.iloc[0]["image_path"]
            has_pothole = int(match.iloc[0]["has_pothole"])
            selected_samples.append((query_id, Path(img_path), category, has_pothole, desc))
        else:
            log.warning("Sample %s not found in manifest!", query_id)

    log.info("Processing %d diagnostic samples for Grad-CAM explainability...", len(selected_samples))

    # Plot figure: 6 rows x 3 columns
    # Col 1: Original Image
    # Col 2: Grad-CAM Overlay
    # Col 3: Isolated Spatial Activation Map
    n_samples = len(selected_samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3.8 * n_samples))

    for i, (query_id, img_path, category, has_pothole, desc) in enumerate(selected_samples):
        orig_img_bgr = cv2.imread(str(img_path))
        if orig_img_bgr is None:
            log.error("Could not read image %s", img_path)
            continue
        orig_img_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = orig_img_rgb.shape

        pil_img = Image.fromarray(orig_img_rgb)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        cam_mask, prob = grad_cam.generate(input_tensor)

        # Resize CAM to original image dimensions
        cam_resized = cv2.resize(cam_mask, (w, h))

        # Colormap heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Superimpose
        blended = np.float32(heatmap) / 255.0 * 0.45 + np.float32(orig_img_rgb) / 255.0 * 0.55
        blended = np.clip(blended, 0, 1)

        pred_label = "Pothole" if prob >= 0.5 else "Normal/Crack"
        is_correct = (prob >= 0.5 and has_pothole == 1) or (prob < 0.5 and has_pothole == 0)
        status_color = "#1b9e77" if is_correct else "#d95f02"

        # Col 1: Original Image
        axes[i, 0].imshow(orig_img_rgb)
        axes[i, 0].set_title(f"[{query_id}] Ground Truth: {category}\n{desc}", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        # Col 2: Grad-CAM Overlay
        axes[i, 1].imshow(blended)
        verdict = f"P(Pothole) = {prob:.3f} -> Predicted: {pred_label}"
        axes[i, 1].set_title(f"Grad-CAM Attention Overlay\n{verdict}", fontsize=10, fontweight="bold", color=status_color)
        axes[i, 1].axis("off")

        # Col 3: Isolated Heatmap
        im3 = axes[i, 2].imshow(cam_resized, cmap="jet", vmin=0, vmax=1)
        axes[i, 2].set_title(f"ResNet-18 Layer4 Spatial Saliency\nPeak Activation: {cam_mask.max():.2f}", fontsize=10)
        axes[i, 2].axis("off")

    plt.suptitle("Explainable AI Diagnosis: ResNet-18 Grad-CAM on Potholes vs. Crack Hard Negatives", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved Grad-CAM explainability visualization to %s", args.out_path)


if __name__ == "__main__":
    main()
