"""HOG feature extraction for structural edges and pothole cavity boundaries."""
from typing import Tuple

import cv2
import numpy as np
from skimage.feature import hog


def extract_hog_features(
    img_bgr: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (16, 16),
    cells_per_block: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Extracts HOG descriptor over grayscale image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=False,
    )
    return features.astype(np.float32)
