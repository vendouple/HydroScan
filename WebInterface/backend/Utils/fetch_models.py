import os
from pathlib import Path


PLACES365_PROTOTXT_URL = os.environ.get(
    "PLACES365_PROTOTXT_URL",
    "https://raw.githubusercontent.com/CSAILVision/places365/master/deploy_resnet152_places365.prototxt",
)
PLACES365_CAFFE_URL = os.environ.get(
    "PLACES365_CAFFE_URL",
    "http://places2.csail.mit.edu/models_places365/resnet152_places365.caffemodel",
)


def ensure_models(base_dir: str) -> None:
    """
    Ensure required model assets exist. If download fails, print concise manual instructions.

    base_dir: path to WebInterface/backend/Models
    """
    models_dir = Path(base_dir)
    place365_dir = models_dir / "Place365"
    place365_dir.mkdir(parents=True, exist_ok=True)

    prototxt_path = place365_dir / "deploy_resnet152_places365.prototxt"
    caffemodel_path = place365_dir / "resnet152_places365.caffemodel"

    # Lazy import requests to avoid hard dependency during cold imports
    def _download(url: str, dst: Path) -> bool:
        try:
            import requests  # type: ignore

            for attempt in range(3):
                try:
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                    dst.write_bytes(resp.content)
                    return True
                except Exception:
                    if attempt == 2:
                        return False
            return False
        except Exception:
            return False

    if not prototxt_path.exists():
        ok = _download(PLACES365_PROTOTXT_URL, prototxt_path)
        if not ok:
            print(
                f"[HydroScan] Could not download prototxt automatically. Please download manually:\n"
                f"  {PLACES365_PROTOTXT_URL}\n"
                f"and place it at: {prototxt_path}"
            )

    if not caffemodel_path.exists():
        ok = _download(PLACES365_CAFFE_URL, caffemodel_path)
        if not ok:
            print(
                f"[HydroScan] Could not download caffemodel automatically. Please download manually:\n"
                f"  {PLACES365_CAFFE_URL}\n"
                f"and place it at: {caffemodel_path}"
            )
