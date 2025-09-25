from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None
    _HAS_CV2 = False

REFERENCE_COLOR = np.array([140, 180, 200], dtype=np.float32)  # light aqua baseline


def _to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _edge_density(gray: np.ndarray) -> float:
    if _HAS_CV2:
        edges = cv2.Canny(gray, 50, 150)
    else:
        edges = np.array(Image.fromarray(gray).filter(ImageFilter.FIND_EDGES))
    density = float(np.count_nonzero(edges)) / (edges.size) * 100.0
    return float(np.clip(density, 0.0, 100.0))


def _sharpness_variance(gray: np.ndarray) -> float:
    if _HAS_CV2:
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    else:
        gy, gx = np.gradient(gray.astype(np.float32))
        lap_var = np.var(gx) + np.var(gy)
    # map variance (clarity) to 0-100 turbidity proxy (inverse relation)
    clarity = float(np.clip((lap_var / 1200.0) * 100.0, 0.0, 100.0))
    turbidity = float(np.clip(100.0 - clarity, 0.0, 100.0))
    return turbidity


def _color_deviation(rgb: np.ndarray) -> float:
    diff = np.linalg.norm(rgb.astype(np.float32) - REFERENCE_COLOR, axis=2)
    deviation = float(np.clip(diff.mean() / 3.0, 0.0, 100.0))
    return deviation


def _foam_proxy(rgb: np.ndarray) -> float:
    mean_channel = rgb.mean(axis=2)
    std_channel = rgb.std(axis=2)
    foam_mask = (mean_channel > 220) & (std_channel < 18)
    ratio = float(foam_mask.mean() * 100.0)
    return float(np.clip(ratio, 0.0, 100.0))


def compute_visual_metrics(img: Image.Image) -> Dict[str, float]:
    """Compute heuristic visual metrics used by scoring pipeline."""

    rgb = _to_rgb_array(img)
    gray = rgb if rgb.ndim == 2 else rgb[:, :, 0] * 0
    if rgb.ndim == 3:
        if _HAS_CV2:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = np.array(img.convert("L"))

    edge_density = _edge_density(gray)
    turbidity_proxy = _sharpness_variance(gray)
    color_deviation = _color_deviation(rgb)
    foam_proxy = _foam_proxy(rgb)

    return {
        "turbidity_proxy": turbidity_proxy,
        "color_deviation": color_deviation,
        "foam_proxy": foam_proxy,
        "edge_density": edge_density,
    }
