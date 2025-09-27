from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from PIL import Image

try:
    from rfdetr import RFDETRMedium
    from rfdetr.util.coco_classes import COCO_CLASSES

    _HAS_RFDETR = True
except Exception:
    RFDETRMedium = None
    COCO_CLASSES = []
    _HAS_RFDETR = False


class RFDETRAdapter:
    """Adapter around the official rfdetr package returning unified detection dicts."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        optimize: bool = True,
        model_name: Optional[str] = None,
    ) -> None:
        self.device = device or os.environ.get("RFDETR_DEVICE")
        models_root_env = os.environ.get("HYDROSCAN_MODELS_DIR")
        if models_root_env:
            models_root = Path(models_root_env)
        else:
            models_root = Path(__file__).resolve().parents[1] / "Models"
        models_root.mkdir(parents=True, exist_ok=True)

        cache_root = models_root / "RFDETR"
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_root = cache_root.resolve()
        print(f"[RFDETR] Cache directory: {cache_root}")

        os.environ["RFD_MODEL_CACHE"] = str(cache_root)
        os.environ["RFDETR_CACHE_DIR"] = str(cache_root)
        os.environ["HF_HOME"] = str(cache_root)
        os.environ["TORCH_HOME"] = str(cache_root)

        self.checkpoint_path = checkpoint_path or os.environ.get("RFDETR_CHECKPOINT")
        requested_variant = (
            model_name or os.environ.get("RFDETR_MODEL_NAME") or "rf_detr_m"
        )
        variant_key = str(requested_variant).lower()
        variant_map = {
            "rf_detr_m": RFDETRMedium,
            "rf-detr-m": RFDETRMedium,
            "rf_detr_medium": RFDETRMedium,
            "medium": RFDETRMedium,
            "m": RFDETRMedium,
        }
        model_cls = variant_map.get(variant_key, RFDETRMedium)
        if model_cls is None:
            print(f"[HydroScan] RF-DETR variant '{requested_variant}' is unavailable")
            return
        self.model_variant = getattr(model_cls, "size", variant_key)
        print(
            f"[RFDETR] Requested variant '{requested_variant}' -> using {self.model_variant}"
        )
        self.model_name = variant_key
        self._class_names: Sequence[str] = list(COCO_CLASSES)
        self.model = None

        if not _HAS_RFDETR:
            print("[HydroScan] rfdetr package not available; RF-DETR adapter disabled.")
            return

        kwargs: Dict[str, Any] = {}
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            kwargs["checkpoint_path"] = self.checkpoint_path

        # Always pass cache_dir to ensure models download to correct location
        kwargs["cache_dir"] = str(cache_root)
        print(f"[RFDETR] Initializing {model_cls.__name__} with kwargs: {kwargs}")

        # Change to the cache directory during model initialization to force downloads there
        original_cwd = os.getcwd()
        try:
            os.chdir(cache_root)
            model = model_cls(**kwargs)
            if hasattr(model, "to") and self.device:
                try:
                    model.to(self.device)
                except Exception:
                    pass
            if optimize and hasattr(model, "optimize_for_inference"):
                optimized = model.optimize_for_inference()
                if optimized is not None:
                    model = optimized
            self.model = model
        except Exception as exc:
            print(f"[HydroScan] Failed to initialize RF-DETR model: {exc}")
            self.model = None
        finally:
            os.chdir(original_cwd)

    # ------------------------------------------------------------------
    def _convert_predictions(self, detections: Any) -> List[Dict[str, Any]]:
        dets: List[Dict[str, Any]] = []
        if detections is None:
            return dets

        # Some versions wrap data in a `predictions` attribute (supervision integration)
        detections = getattr(detections, "predictions", detections)

        xyxy = getattr(detections, "xyxy", None)
        if xyxy is None:
            xyxy = getattr(detections, "boxes", None)
        class_ids = getattr(detections, "class_id", None)
        if class_ids is None:
            class_ids = getattr(detections, "class_ids", None)
        confidences = getattr(detections, "confidence", None)
        if confidences is None:
            confidences = getattr(detections, "confidences", None)

        if xyxy is None or class_ids is None or confidences is None:
            return dets

        # Convert to list-friendly numpy arrays where possible
        try:
            import numpy as np

            if hasattr(xyxy, "cpu"):
                xyxy = xyxy.cpu().numpy()
            else:
                xyxy = np.asarray(xyxy)

            if hasattr(class_ids, "cpu"):
                class_ids = class_ids.cpu().numpy()
            else:
                class_ids = np.asarray(class_ids)

            if hasattr(confidences, "cpu"):
                confidences = confidences.cpu().numpy()
            else:
                confidences = np.asarray(confidences)
        except Exception:
            pass

        length = min(len(class_ids), len(confidences))
        for idx in range(length):
            try:
                bbox = xyxy[idx].tolist()
            except Exception:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            class_id = int(class_ids[idx])
            score = float(confidences[idx])
            class_name = (
                self._class_names[class_id]
                if 0 <= class_id < len(self._class_names)
                else str(class_id)
            )
            dets.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "class_name": class_name,
                    "score": score,
                    "source": "rfdetr",
                }
            )
        return dets

    # ------------------------------------------------------------------
    def predict(
        self, image: Image.Image, threshold: float = 0.3  # Lower threshold
    ) -> List[Dict[str, Any]]:
        if self.model is None:
            print("[RFDETR] Model not loaded")
            return []
        try:
            print(
                f"[RFDETR] Running inference with threshold {threshold} on image size {image.size}"
            )
            detections = self.model.predict(image, threshold=threshold)
            print(
                f"[RFDETR] Raw detections type: {type(detections)}, content: {detections}"
            )

            # RFDETRBase.predict returns a list; take first element for single-image inference
            if isinstance(detections, list) and detections:
                print(f"[RFDETR] Taking first detection from list of {len(detections)}")
                detections = detections[0]

            print(f"[RFDETR] Processing detections: {type(detections)}")
            if hasattr(detections, "xyxy"):
                print(f"[RFDETR] Has xyxy: {getattr(detections, 'xyxy', None)}")
            if hasattr(detections, "class_id"):
                print(f"[RFDETR] Has class_id: {getattr(detections, 'class_id', None)}")
            if hasattr(detections, "confidence"):
                print(
                    f"[RFDETR] Has confidence: {getattr(detections, 'confidence', None)}"
                )

            results = self._convert_predictions(detections)
            print(f"[RFDETR] Found {len(results)} detections after conversion")
            return results
        except Exception as exc:
            print(f"[HydroScan] RF-DETR inference failed: {exc}")
            import traceback

            traceback.print_exc()
            return []

    def predict_batch(
        self, images: Iterable[Image.Image], threshold: float = 0.5
    ) -> List[List[Dict[str, Any]]]:
        batches = list(images)
        if self.model is None:
            return [[] for _ in batches]
        outputs: List[List[Dict[str, Any]]] = []
        for img in batches:
            outputs.append(self.predict(img, threshold=threshold))
        return outputs
