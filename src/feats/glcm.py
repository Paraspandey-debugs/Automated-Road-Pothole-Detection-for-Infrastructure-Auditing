"""GLCM texture properties: contrast, dissimilarity, homogeneity, energy, correlation."""
from typing import Sequence

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_glcm_features(
    img_bgr: np.ndarray,
    distances: Sequence[int] = (1, 3),
    angles: Sequence[float] = (0.0, 0.785398, 1.570796, 2.356194),
) -> np.ndarray:
    """Computes mean and standard deviation of Haralick/GLCM features."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Quantize to 64 gray levels for fast, invariant GLCM computation
    gray_quantized = (gray // 4).astype(np.uint8)

    glcm = graycomatrix(
        gray_quantized,
        distances=list(distances),
        angles=list(angles),
        levels=64,
        symmetric=True,
        normed=True,
    )

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    feats = []
    for prop in props:
        vals = graycoprops(glcm, prop).ravel()
        feats.extend([float(np.mean(vals)), float(np.std(vals))])

    return np.array(feats, dtype=np.float32)
