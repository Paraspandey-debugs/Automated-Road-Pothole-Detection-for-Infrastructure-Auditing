"""Color feature extraction: HSV channel statistics (mean, std, skew) and 3D histograms."""
from typing import Tuple

import cv2
import numpy as np
from scipy.stats import skew


def extract_color_features(img_bgr: np.ndarray, bins: Tuple[int, int, int] = (16, 16, 16)) -> np.ndarray:
    """
    Extracts color moments (mean, standard deviation, skewness) and
    normalized 3D color histogram in HSV color space.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Channel statistical moments
    stats = []
    for c in range(3):
        channel = hsv[:, :, c].ravel().astype(np.float32)
        stats.extend([float(np.mean(channel)), float(np.std(channel)), float(skew(channel))])

    # Normalized 3D HSV histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)

    return np.concatenate([np.array(stats, dtype=np.float32), hist])
