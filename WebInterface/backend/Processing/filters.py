from typing import List, Tuple
from PIL import Image


def generate_variants(img: Image.Image) -> List[Tuple[str, Image.Image]]:
    """
    Return list of (variant_name, image). Placeholder returns only original.
    """
    return [("original", img)]
