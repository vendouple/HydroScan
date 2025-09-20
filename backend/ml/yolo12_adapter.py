from .detector_base import DetectorAdapter, Detection
from typing import List
import numpy as np
import onnxruntime as ort
import cv2
import os


class YOLO12Adapter(DetectorAdapter):
    """
    ONNXRuntime-based inference for YOLOv12 exported models.
    Expect standard YOLO ONNX outputs (boxes in xyxy + scores + class probs).
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError("YOLOv12 requires an exported ONNX model path")
        self.model_path = model_path
        self.device = device
        providers = (
            ["CPUExecutionProvider"]
            if device.startswith("cpu")
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [N,3,H,W]

    def _prep(self, img: np.ndarray):
        _, _, H, W = [None, None] + self.input_shape[2:]
        H = int(H or 640)
        W = int(W or 640)
        im = cv2.resize(img[:, :, ::-1], (W, H))  # RGB->BGR then resize
        im = im[:, :, ::-1]  # back to RGB
        im = im.astype(np.float32) / 255.0
        im = np.transpose(im, (2, 0, 1))
        im = np.expand_dims(im, 0)
        return im

    def predict(self, images: List[np.ndarray], conf: float = 0.25, iou: float = 0.45):
        outs = []
        for img in images:
            inp = self._prep(img)
            outputs = self.session.run(None, {self.input_name: inp})
            # This part depends on the exported model; we'll handle a common format: [num,6] = x1,y1,x2,y2,conf,cls
            dets = []
            for out in outputs:
                arr = np.array(out)
                if arr.ndim == 2 and arr.shape[1] >= 6:
                    for row in arr:
                        x1, y1, x2, y2, sc, cls = row[:6]
                        if sc >= conf:
                            dets.append(
                                Detection(
                                    (float(x1), float(y1), float(x2), float(y2)),
                                    str(int(cls)),
                                    float(sc),
                                )
                            )
            outs.append(dets)
        return outs

    def model_info(self) -> dict:
        return {
            "backend": "yolo12",
            "model": os.path.basename(self.model_path),
            "device": self.device,
        }
