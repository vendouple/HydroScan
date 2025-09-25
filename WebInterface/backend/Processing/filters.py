from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    _HAS_CV2 = False


MAX_DEFAULT_VARIANTS = int(os.environ.get("HYDROSCAN_MAX_VARIANTS", "6"))


def _to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _from_numpy(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _auto_white_balance(arr: np.ndarray) -> np.ndarray:
    arr_f = arr.astype(np.float32)
    mean_channels = arr_f.mean(axis=(0, 1)) + 1e-6
    gray_mean = mean_channels.mean()
    scale = gray_mean / mean_channels
    balanced = arr_f * scale
    return np.clip(balanced, 0, 255).astype(np.uint8)


def _apply_clahe(arr: np.ndarray) -> np.ndarray:
    if _HAS_CV2:
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # Pillow fallback using histogram equalization
    pil_img = Image.fromarray(arr)
    return np.array(ImageOps.equalize(pil_img))


def _denoise(arr: np.ndarray) -> np.ndarray:
    if _HAS_CV2:
        return cv2.fastNlMeansDenoisingColored(arr, None, 10, 10, 7, 21)
    # Pillow fallback: median + slight blur
    pil_img = Image.fromarray(arr)
    return np.array(
        pil_img.filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.SMOOTH)
    )


def _sharpen(arr: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(arr)
    return np.array(
        pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    )


def _gamma(arr: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(
        np.uint8
    )
    return cv2.LUT(arr, table) if _HAS_CV2 else table[arr]


def _deglare(arr: np.ndarray) -> np.ndarray:
    if _HAS_CV2:
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        v = np.clip(v * 0.9, 0, 255).astype(np.uint8)
        suppressed = cv2.merge((h, s, v))
        return cv2.cvtColor(suppressed, cv2.COLOR_HSV2RGB)
    pil_img = Image.fromarray(arr)
    enhancer = ImageEnhance.Brightness(pil_img)
    darker = enhancer.enhance(0.9)
    return np.array(darker.filter(ImageFilter.SMOOTH))


def generate_variants(
    img: Image.Image, max_variants: int | None = None
) -> List[Tuple[str, Image.Image]]:
    """Generate filtered variants for downstream analysis."""

    source = img.convert("RGB")
    base_arr = _to_numpy(source)

    variants: List[Tuple[str, Image.Image]] = [("original", source)]

    operations = [
        ("auto_white_balance", lambda: _from_numpy(_auto_white_balance(base_arr))),
        ("contrast_stretch", lambda: _from_numpy(_apply_clahe(base_arr))),
        ("denoise", lambda: _from_numpy(_denoise(base_arr))),
        ("sharpen", lambda: _from_numpy(_sharpen(base_arr))),
        ("gamma_1.2", lambda: _from_numpy(_gamma(base_arr, gamma=1.2))),
        ("deglare", lambda: _from_numpy(_deglare(base_arr))),
    ]

    limit = max_variants or MAX_DEFAULT_VARIANTS
    for name, fn in operations:
        if len(variants) >= limit:
            break
        try:
            processed = fn()
            variants.append((name, processed))
        except Exception as exc:  # pragma: no cover - best effort resilience
            print(f"[HydroScan] Filter '{name}' failed: {exc}")

    return variants
