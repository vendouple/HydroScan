from typing import List, Dict
from PIL import Image


class RFDETRAdapter:
    def __init__(self, checkpoint_path: str | None = None):
        self.checkpoint_path = checkpoint_path

    def predict(self, pil_image: Image.Image, threshold: float = 0.5) -> List[Dict]:
        return []
