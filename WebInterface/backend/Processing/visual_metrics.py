from typing import Dict
from PIL import Image


def compute_visual_metrics(img: Image.Image) -> Dict[str, float]:
    """Compute turbidity proxy, color deviation, foam proxy. Placeholder values."""
    return {
        "turbidity_proxy": 0.0,
        "color_deviation": 0.0,
        "foam_proxy": 0.0,
        "edge_density": 0.0,
    }
