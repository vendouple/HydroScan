from typing import List, Dict, Any, Optional
import os
from PIL import Image


class InModelAdapter:
    """
    Wrapper around Ultralytics YOLO (v11 and v12 compatible) for in-house models.

    Returns unified detections:
    {"bbox": [x1,y1,x2,y2], "class_id": int, "class_name": str, "score": float, "source": "inmodel"}
    """

    def __init__(
        self, weights_path: Optional[str] = None, device: Optional[str] = None
    ):
        self.weights_path = weights_path
        self.device = device or os.environ.get(
            "YOLO_DEVICE"
        )  # e.g., "cpu" or "0"/"cuda:0"
        self.model = None

        if weights_path and os.path.exists(weights_path):
            try:
                from ultralytics import YOLO  # type: ignore

                self.model = YOLO(weights_path)
                # Optional: move to device if specified
                if self.device:
                    # YOLO API uses .to() via model.model
                    try:
                        self.model.to(self.device)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[HydroScan] Failed to load YOLO model at {weights_path}: {e}")
        else:
            print(f"[HydroScan] InModelAdapter: weights not found at {weights_path}")

    def _result_to_detections(self, result) -> List[Dict[str, Any]]:
        dets: List[Dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or {}
        if boxes is None:
            return dets
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            # fallback (should be rare)
            return dets

        for i in range(len(cls)):
            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
            score = float(conf[i])
            cid = int(cls[i])
            cname = names.get(cid, str(cid))
            dets.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cid,
                    "class_name": cname,
                    "score": score,
                    "source": "inmodel",
                }
            )
        return dets

    def predict(
        self, images: List[Image.Image], conf: float = 0.25, iou: float = 0.45
    ) -> List[List[Dict[str, Any]]]:
        if self.model is None:
            return [[] for _ in images]

        # Run batched inference; Ultralytics supports list[ndarray|PIL]
        try:
            results = self.model.predict(
                source=images, conf=conf, iou=iou, verbose=False
            )
        except Exception as e:
            print(f"[HydroScan] YOLO inference failed: {e}")
            return [[] for _ in images]

        all_dets: List[List[Dict[str, Any]]] = []
        for r in results:
            dets = self._result_to_detections(r)
            all_dets.append(dets)
        return all_dets
