from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except (
    Exception
):  # pragma: no cover - optional dependency for environments without OpenCV
    cv2 = None  # type: ignore
    _HAS_CV2 = False


@dataclass(slots=True)
class _Category:
    index: int
    path: str
    label: str


class Place365Adapter:
    """Places365 scene classifier using official Caffe prototxt + caffemodel."""

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.place_dir = self.models_dir / "Place365"
        self.prototxt = self.place_dir / "deploy_resnet152_places365.prototxt"
        self.caffemodel = self.place_dir / "resnet152_places365.caffemodel"
        self.categories_file = self.place_dir / "categories_places365.txt"
        self.io_file = self.place_dir / "IO_places365.txt"

        self.net = None
        self.categories: List[_Category] = []
        self.scene_lookup: Dict[str, str] = {}

        self._load_metadata()
        self._load_network()

    # ------------------------------------------------------------------
    def _load_network(self) -> None:
        if not _HAS_CV2:
            print("[HydroScan] OpenCV is not installed; Places365 disabled.")
            return
        if not self.prototxt.exists() or not self.caffemodel.exists():
            print(
                "[HydroScan] Places365 assets missing (prototxt / caffemodel). "
                "Run fetch_models or download manually."
            )
            return
        try:
            self.net = cv2.dnn.readNetFromCaffe(
                str(self.prototxt), str(self.caffemodel)
            )
        except Exception as exc:
            print(f"[HydroScan] Failed to load Places365 Caffe model: {exc}")
            self.net = None

    # ------------------------------------------------------------------
    def _load_metadata(self) -> None:
        self.categories = []
        if self.categories_file.exists():
            with self.categories_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_path, idx_str = line.rsplit(" ", 1)
                        idx = int(idx_str)
                    except ValueError:
                        raw_path = line
                        idx = len(self.categories)
                    clean_path = self._clean_path(raw_path)
                    label = self._pretty_label(clean_path)
                    self._upsert_category(idx, clean_path, label)

        if self.io_file.exists():
            with self.io_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_path, io_str = line.rsplit(" ", 1)
                        io_code = int(io_str)
                    except ValueError:
                        continue
                    clean_path = self._clean_path(raw_path)
                    scene = (
                        "indoor"
                        if io_code == 1
                        else "outdoor" if io_code == 2 else "unknown"
                    )
                    self.scene_lookup[clean_path] = scene

        # Ensure lists are index aligned
        self.categories.sort(key=lambda c: c.index)

    # ------------------------------------------------------------------
    @staticmethod
    def _clean_path(raw_path: str) -> str:
        path = raw_path.strip()
        if path.startswith("/"):
            path = path[1:]
        # Drop the grouping prefix (e.g. "a/") if present
        if "/" in path:
            path = path.split("/", 1)[1]
        return path

    @staticmethod
    def _pretty_label(clean_path: str) -> str:
        parts = [segment.replace("_", " ") for segment in clean_path.split("/")]
        return " / ".join(parts)

    def _upsert_category(self, idx: int, clean_path: str, label: str) -> None:
        for cat in self.categories:
            if cat.index == idx:
                return
        self.categories.append(_Category(index=idx, path=clean_path, label=label))

    # ------------------------------------------------------------------
    def classify(self, pil_image: Image.Image) -> Dict[str, object]:
        if self.net is None:
            return {"scene": "unknown", "label": "unknown", "confidence": 0.0}

        image = pil_image.convert("RGB")
        np_img = np.array(image, dtype=np.float32)
        try:
            blob = cv2.dnn.blobFromImage(
                np_img,
                scalefactor=1.0,
                size=(224, 224),
                mean=[104, 117, 123],
                swapRB=True,
                crop=False,
            )
            self.net.setInput(blob)
            prob = self.net.forward()
            prob = np.squeeze(prob)
            if prob.ndim != 1:
                return {"scene": "unknown", "label": "unknown", "confidence": 0.0}
            # Normalize if softmax layer absent
            if float(np.sum(prob)) <= 0 or np.any(prob < 0):
                exp = np.exp(prob - np.max(prob))
                prob = exp / np.sum(exp)
            idx = int(np.argmax(prob))
            confidence = float(prob[idx])
        except Exception as exc:
            print(f"[HydroScan] Places365 inference failed: {exc}")
            return {"scene": "unknown", "label": "unknown", "confidence": 0.0}

        if 0 <= idx < len(self.categories):
            category = self.categories[idx]
            label = category.label
            scene = self.scene_lookup.get(category.path, "unknown")
        else:
            label = str(idx)
            scene = "unknown"

        return {"scene": scene, "label": label, "confidence": confidence}
