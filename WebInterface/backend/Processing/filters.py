from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    _HAS_CV2 = False


MAX_DEFAULT_VARIANTS = int(os.environ.get("HYDROSCAN_MAX_VARIANTS", "6"))

DEFAULT_SEQUENCE: Sequence[str] = (
    "auto_white_balance",
    "contrast_stretch",
    "denoise",
    "sharpen",
    "gamma_adaptive",
    "deglare",
    "red_enhance",
    "blue_enhance",
    "green_enhance",
    "cyan_filter",
)

FILTER_DISPLAY_NAMES = {
    "original": "Original",
    "auto_white_balance": "White balance",
    "contrast_stretch": "Contrast stretch",
    "denoise": "Denoise",
    "sharpen": "Edge sharpen",
    "gamma_adaptive": "Adaptive gamma correction",
    "gamma_1.2": "Gamma boost 1.2",
    "gamma_1.4": "Gamma boost 1.4",
    "gamma_1.6": "Gamma boost 1.6",
    "deglare": "Highlight suppression",
    "red_enhance": "Red enhancement",
    "blue_enhance": "Blue enhancement",
    "green_enhance": "Green enhancement",
    "cyan_filter": "Cyan filter",
    "yellow_filter": "Yellow filter",
    "magenta_filter": "Magenta filter",
}


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


def _calculate_dynamic_gamma(arr: np.ndarray) -> float:
    """Calculate optimal gamma value based on image characteristics."""
    # Convert to grayscale for analysis
    if len(arr.shape) == 3:
        gray = np.dot(arr[..., :3], [0.2989, 0.587, 0.114])
    else:
        gray = arr

    # Calculate image statistics
    mean_brightness = np.mean(gray) / 255.0
    std_brightness = np.std(gray) / 255.0

    # Histogram analysis for exposure
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist_norm = hist / np.sum(hist)

    # Check for underexposure (peaks in lower third)
    underexposed = np.sum(hist_norm[:85]) > 0.6
    # Check for overexposure (peaks in upper third)
    overexposed = np.sum(hist_norm[170:]) > 0.6

    # Adaptive gamma calculation
    if underexposed and mean_brightness < 0.4:
        # Image is too dark, need higher gamma to brighten
        gamma = 1.6 + (0.4 - mean_brightness) * 0.5
    elif overexposed and mean_brightness > 0.7:
        # Image is too bright, need lower gamma to darken
        gamma = 0.8 - (mean_brightness - 0.7) * 0.3
    elif std_brightness < 0.15:
        # Low contrast, moderate gamma boost
        gamma = 1.3
    else:
        # Normal exposure, slight enhancement
        gamma = 1.2

    # Clamp gamma to reasonable range
    return max(0.5, min(2.0, gamma))


def _gamma(arr: np.ndarray, gamma: float = None) -> np.ndarray:
    if gamma is None:
        gamma = _calculate_dynamic_gamma(arr)

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


def _color_enhance(arr: np.ndarray, channel: str, factor: float = 1.3) -> np.ndarray:
    """Enhance specific color channels for better water detection."""
    enhanced = arr.astype(np.float32)

    if channel == "red":
        enhanced[:, :, 0] *= factor
    elif channel == "green":
        enhanced[:, :, 1] *= factor
    elif channel == "blue":
        enhanced[:, :, 2] *= factor
    elif channel == "cyan":
        # Cyan = high blue + green, low red
        enhanced[:, :, 0] *= 0.7  # reduce red
        enhanced[:, :, 1] *= factor  # enhance green
        enhanced[:, :, 2] *= factor  # enhance blue
    elif channel == "yellow":
        # Yellow = high red + green, low blue
        enhanced[:, :, 0] *= factor  # enhance red
        enhanced[:, :, 1] *= factor  # enhance green
        enhanced[:, :, 2] *= 0.7  # reduce blue
    elif channel == "magenta":
        # Magenta = high red + blue, low green
        enhanced[:, :, 0] *= factor  # enhance red
        enhanced[:, :, 1] *= 0.7  # reduce green
        enhanced[:, :, 2] *= factor  # enhance blue

    return np.clip(enhanced, 0, 255).astype(np.uint8)


def _apply_operation(
    img: Image.Image,
    operation: str,
    base_arr: np.ndarray | None = None,
) -> Image.Image:
    if operation == "original":
        return img.convert("RGB")

    if base_arr is None:
        base_arr = _to_numpy(img)

    if operation == "auto_white_balance":
        return _from_numpy(_auto_white_balance(base_arr))
    if operation == "contrast_stretch":
        return _from_numpy(_apply_clahe(base_arr))
    if operation == "denoise":
        return _from_numpy(_denoise(base_arr))
    if operation == "sharpen":
        return _from_numpy(_sharpen(base_arr))
    if operation.startswith("gamma"):
        if operation == "gamma_adaptive":
            return _from_numpy(_gamma(base_arr, gamma=None))  # Use dynamic calculation
        else:
            try:
                gamma_value = float(operation.split("_")[1])
            except (IndexError, ValueError):
                gamma_value = 1.2
            return _from_numpy(_gamma(base_arr, gamma=gamma_value))
    if operation == "deglare":
        return _from_numpy(_deglare(base_arr))
    if operation == "red_enhance":
        return _from_numpy(_color_enhance(base_arr, "red"))
    if operation == "blue_enhance":
        return _from_numpy(_color_enhance(base_arr, "blue"))
    if operation == "green_enhance":
        return _from_numpy(_color_enhance(base_arr, "green"))
    if operation == "cyan_filter":
        return _from_numpy(_color_enhance(base_arr, "cyan"))
    if operation == "yellow_filter":
        return _from_numpy(_color_enhance(base_arr, "yellow"))
    if operation == "magenta_filter":
        return _from_numpy(_color_enhance(base_arr, "magenta"))

    raise ValueError(f"Unknown filter operation '{operation}'")


def generate_variants(
    img: Image.Image,
    max_variants: int | None = None,
    operations: Iterable[str] | None = None,
    include_original: bool = True,
) -> List[Tuple[str, Image.Image]]:
    """Generate filtered variants for downstream analysis.

    Parameters
    ----------
    img:
        Source image to transform.
    max_variants:
        Maximum number of variants to return (including original if applicable).
    operations:
        Explicit ordered list of operations to apply. When omitted, the
        `DEFAULT_SEQUENCE` is used.
    include_original:
        Whether to include the unmodified image as the first variant.
    """

    source = img.convert("RGB")
    base_arr = _to_numpy(source)

    sequence = list(operations) if operations is not None else list(DEFAULT_SEQUENCE)
    limit = max_variants or MAX_DEFAULT_VARIANTS

    variants: List[Tuple[str, Image.Image]] = []
    if include_original:
        variants.append(("original", source))

    for name in sequence:
        if len(variants) >= limit:
            break
        try:
            processed = _apply_operation(source, name, base_arr)
        except Exception as exc:  # pragma: no cover - best effort resilience
            print(f"[HydroScan] Filter '{name}' failed: {exc}")
            continue
        variants.append((name, processed))

    return variants


def get_filter_display_name(operation: str) -> str:
    return FILTER_DISPLAY_NAMES.get(operation, operation.replace("_", " ").title())


def get_adaptive_filter_sequence(img: Image.Image) -> List[str]:
    """Generate an adaptive filter sequence based on image characteristics."""
    base_arr = _to_numpy(img)

    # Always start with basic corrections
    sequence = ["auto_white_balance", "contrast_stretch"]

    # Calculate image statistics for adaptive selection
    gray = np.dot(base_arr[..., :3], [0.2989, 0.587, 0.114])
    mean_brightness = np.mean(gray) / 255.0
    noise_estimate = np.std(gray) / 255.0

    # Add filters based on image quality assessment
    if noise_estimate > 0.25:  # High noise
        sequence.append("denoise")

    # Always add sharpening for water detection
    sequence.append("sharpen")

    # Add adaptive gamma based on exposure
    sequence.append("gamma_adaptive")

    # Add glare reduction if image is bright
    if mean_brightness > 0.6:
        sequence.append("deglare")

    # Add color enhancement filters for water detection
    # Focus on blue/cyan for water, but include others for various conditions
    sequence.extend(["blue_enhance", "cyan_filter", "green_enhance"])

    # Add additional filters for challenging conditions
    if mean_brightness < 0.3 or mean_brightness > 0.8:  # Poor exposure
        sequence.extend(["red_enhance", "yellow_filter"])

    return sequence
