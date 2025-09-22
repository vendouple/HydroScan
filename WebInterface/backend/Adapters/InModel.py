from typing import List, Dict
from PIL import Image


class InModelAdapter:
    def __init__(self, weights_path: str | None = None):
        self.weights_path = weights_path

    def predict(
        self, images: List[Image.Image], conf: float = 0.25, iou: float = 0.45
    ) -> List[List[Dict]]:
        return [[] for _ in images]
