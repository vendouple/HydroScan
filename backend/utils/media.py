from __future__ import annotations
import cv2
import numpy as np
from typing import Iterable, List, Tuple

MAX_VIDEO_SECONDS = 60
TARGET_FPS = 30


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img[:, :, ::-1]  # BGR->RGB


def sample_video_frames(
    path: str, max_seconds: int = MAX_VIDEO_SECONDS, target_fps: int = TARGET_FPS
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = frame_count / max(fps, 1e-6)

    # Limit duration
    if duration_s > max_seconds:
        max_frames = int(max_seconds * fps)
    else:
        max_frames = frame_count

    # Adaptive sampling: stride to target FPS or lower
    stride = max(int(round(fps / target_fps)) if fps > target_fps else 1, 1)
    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame[:, :, ::-1])
        idx += 1
        if idx >= max_frames:
            break
    cap.release()
    return frames


def apply_filters(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Generate filtered variants: original, denoise, CLAHE, deglare-ish."""
    variants: List[Tuple[str, np.ndarray]] = [("original", img)]
    # Denoise
    dn = cv2.fastNlMeansDenoisingColored(
        (img[:, :, ::-1]).astype(np.uint8), None, 3, 3, 7, 21
    )[:, :, ::-1]
    variants.append(("denoise", dn))
    # CLAHE on L channel
    lab = cv2.cvtColor(img[:, :, ::-1], cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    clab = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(clab, cv2.COLOR_LAB2BGR)[:, :, ::-1]
    variants.append(("clahe", clahe_img))
    # Simple deglare via bilateral filter
    deglare = cv2.bilateralFilter((img[:, :, ::-1]).astype(np.uint8), 9, 75, 75)[
        :, :, ::-1
    ]
    variants.append(("deglare", deglare))
    return variants
