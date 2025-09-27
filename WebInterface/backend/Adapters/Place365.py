from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import URLError
from urllib.request import urlopen

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

        self.place_dir.mkdir(parents=True, exist_ok=True)
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
        self._ensure_metadata_files()

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
        else:
            # Add common scene categories when file is missing
            print("[Place365] Category file missing, using default categories")
            self._add_default_categories()

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
        if not self.categories:
            self._add_default_categories()

    # ------------------------------------------------------------------
    def _ensure_metadata_files(self) -> None:
        """Download official metadata files when absent."""

        sources = [
            (
                self.categories_file,
                "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt",
            ),
            (
                self.io_file,
                "https://raw.githubusercontent.com/CSAILVision/places365/master/IO_places365.txt",
            ),
        ]

        for target, url in sources:
            if target.exists():
                continue
            try:
                with urlopen(url, timeout=10) as response:  # nosec B310
                    data = response.read()
                    target.write_bytes(data)
                    print(f"[Place365] Downloaded metadata file: {target.name}")
            except (URLError, OSError, TimeoutError) as exc:
                print(
                    f"[Place365] Failed to download {target.name} ({exc}); falling back to defaults"
                )

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

    def _add_default_categories(self) -> None:
        """Add common scene categories when category files are missing."""
        default_categories = [
            (0, "airplane_cabin", "Airplane Cabin"),
            (1, "airport_terminal", "Airport Terminal"),
            (2, "alley", "Alley"),
            (3, "amphitheater", "Amphitheater"),
            (4, "amusement_arcade", "Amusement Arcade"),
            (5, "amusement_park", "Amusement Park"),
            (50, "cottage", "Cottage"),
            (51, "village", "Village"),
            (52, "countryside", "Countryside"),
            (53, "farmhouse", "Farmhouse"),
            (107, "bottle_storage", "Bottle Storage"),  # Index 107 from your test
            (108, "bow_window/indoor", "Bow Window Indoor"),
            (109, "bowling_alley", "Bowling Alley"),
            (110, "boxing_ring", "Boxing Ring"),
        ]

        for idx, path, label in default_categories:
            self._upsert_category(idx, path, label)
            # Set default scene type
            if "indoor" in path or "cabin" in path or "terminal" in path:
                self.scene_lookup[path] = "indoor"
            else:
                self.scene_lookup[path] = "outdoor"

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

        # Find category by index
        category = None
        for cat in self.categories:
            if cat.index == idx:
                category = cat
                break

        if category:
            label = category.label
            scene = self.scene_lookup.get(category.path, "outdoor")
        else:
            print(
                f"[Place365] Category {idx} not found in {len(self.categories)} loaded categories"
            )
            # Fallback - try to add this index as a generic category
            generic_label = f"Scene {idx}"
            self._upsert_category(idx, f"generic_scene_{idx}", generic_label)
            label = generic_label
            scene = "outdoor"

        return {"scene": scene, "label": label, "confidence": confidence}
