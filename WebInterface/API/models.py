from fastapi import APIRouter
import os
from pathlib import Path

from WebInterface.backend.Utils.fetch_models import get_model_status

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "WebInterface" / "backend" / "Models"


@router.get("/models/status")
def models_status() -> dict:
    base_dir = os.environ.get("HYDROSCAN_MODELS_DIR", str(MODELS_DIR))
    status = get_model_status(base_dir)
    return {"models_dir": base_dir, "assets": status}
