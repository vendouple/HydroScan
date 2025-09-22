from typing import Dict
from PIL import Image


class Place365Adapter:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir

    def classify(self, pil_image: Image.Image) -> Dict[str, object]:
        return {"scene": "unknown", "label": "unknown", "confidence": 0.0}
