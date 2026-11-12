"""LBP texture features capturing road roughness and asphalt degradation."""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def extract_lbp_features(
    img_bgr: np.ndarray,
    num_points: int = 16,
    radius: int = 2,
    method: str = "uniform",
) -> np.ndarray:
    """Extracts uniform rotation-invariant LBP histogram."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=num_points, R=radius, method=method)

    # Uniform LBP has (num_points + 2) distinct pattern bins
    n_bins = int(num_points + 2)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)
