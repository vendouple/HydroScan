from .detector_base import DetectorAdapter, Detection
from typing import List
import numpy as np
import os

# Ultralytics is AGPL-3.0. Ensure usage aligns with your licensing.
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None


class YOLO11Adapter(DetectorAdapter):
    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        if YOLO is None:
            raise RuntimeError("ultralytics is required for YOLO11 backend")
        self.model_path = model_path or os.getenv("YOLO_MODEL", "yolo11n.pt")
        self.device = device
        self.model = YOLO(self.model_path)
        # Attempt to set device
        try:
            self.model.to(self.device)
        except Exception:
            pass

    def predict(self, images: List[np.ndarray], conf: float = 0.25, iou: float = 0.45):
        results = self.model.predict(
            images, conf=conf, iou=iou, verbose=False, device=self.device
        )
        out = []
        names = self.model.names
        for r in results:
            dets = []
            if r.boxes is not None:
                for b in r.boxes:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                    cls_id = (
                        int(b.cls[0].item())
                        if hasattr(b, "cls") and b.cls is not None
                        else -1
                    )
                    cls_name = (
                        names.get(cls_id, str(cls_id))
                        if isinstance(names, dict)
                        else str(cls_id)
                    )
                    score = (
                        float(b.conf[0].item())
                        if hasattr(b, "conf") and b.conf is not None
                        else 0.0
                    )
                    dets.append(Detection((x1, y1, x2, y2), cls_name, score))
            out.append(dets)
        return out

    def model_info(self) -> dict:
        return {
            "backend": "yolo11",
            "model": os.path.basename(self.model_path),
            "device": self.device,
            "version": getattr(self.model, "version", "unknown"),
        }
