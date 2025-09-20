from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    bbox_xyxy: Tuple[float, float, float, float]
    cls: str
    score: float
    track_id: Optional[int] = None


class DetectorAdapter:
    def predict(
        self, images: List[np.ndarray], conf: float = 0.25, iou: float = 0.45
    ) -> List[List[Detection]]:
        raise NotImplementedError

    def model_info(self) -> dict:
        raise NotImplementedError
