"""FastAPI router providing debugging utilities for model inspection.

This endpoint powers the Model Lab page, allowing operators to run individual
models against a supplied image and inspect detections or failure traces.
"""

from __future__ import annotations

import base64
import traceback
from io import BytesIO
from typing import Callable, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from .analyze import (
    _annotate_detections,
    _ensure_image,
    _load_inmodel_adapter,
    _load_place_adapter,
    _load_rfdetr_adapter,
    _serialize_for_json,
)

router = APIRouter(prefix="/debug", tags=["debug"])

SUPPORTED_MODELS: Dict[str, Dict[str, Callable]] = {
    "place365": {
        "label": "Places365 Scene Classifier",
        "loader": _load_place_adapter,
        "predict": lambda adapter, image: adapter.classify(image),
    },
    "rf_detr": {
        "label": "RF-DETR Object Detector",
        "loader": _load_rfdetr_adapter,
        "predict": lambda adapter, image: adapter.predict(image),
    },
    "inmodel": {
        "label": "In-Model Consistency Detector",
        "loader": _load_inmodel_adapter,
        "predict": lambda adapter, image: adapter.predict(image),
    },
}


def _encode_image_preview(image: Image.Image) -> str:
    """Return a base64 JPEG preview suitable for inline rendering."""

    buffer = BytesIO()
    preview = image.convert("RGB")
    preview.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@router.post("/run")
async def run_model_debug(
    model_key: str = Form(..., description="Identifier of the model to run"),
    threshold: float = Form(0.5, description="Optional score threshold"),
    image_file: UploadFile = File(..., description="Image to analyze"),
) -> JSONResponse:
    """Execute a single model for diagnostics and return structured results."""

    model_config = SUPPORTED_MODELS.get(model_key)
    if not model_config:
        raise HTTPException(status_code=400, detail="Unsupported model key")

    try:
        contents = await image_file.read()
        pil_image = Image.open(BytesIO(contents))
        _ensure_image(pil_image)
    except Exception as exc:  # pragma: no cover - validation path
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    response = {
        "status": "ok",
        "model": model_key,
        "model_label": model_config["label"],
        "threshold": threshold,
        "image_preview": _encode_image_preview(pil_image),
    }

    try:
        adapter = model_config["loader"]()
        predictions = model_config["predict"](adapter, pil_image)

        if isinstance(predictions, list):
            detections: List[dict] = predictions
        else:
            detections = [predictions]

        response["detections"] = _serialize_for_json(detections)

        if detections:
            annotated = _annotate_detections(pil_image, detections)
            response["annotated_image"] = _encode_image_preview(annotated)
        else:
            response["message"] = "No detections above threshold"

    except Exception as exc:  # pragma: no cover - runtime diagnostics path
        trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(exc),
                "traceback": trace,
                "model": model_key,
                "model_label": model_config["label"],
            },
        )

    return JSONResponse(status_code=200, content=response)
