from __future__ import annotations

import io
import json
import traceback
import base64
from typing import Any, Dict
from PIL import Image

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/model-test")
async def test_model(
    image: UploadFile = File(...), model: str = Form(...)
) -> Dict[str, Any]:
    """Test individual models with detailed diagnostics and stack traces."""

    try:
        # Load image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        result = {"model": model, "success": False, "error": None, "stack_trace": None}

        if model == "places365":
            result.update(await test_places365(pil_image))
        elif model == "rfdetr":
            result.update(await test_rfdetr(pil_image))
        elif model == "inmodel-classification":
            result.update(await test_inmodel_classification(pil_image))
        elif model == "inmodel-obb":
            result.update(await test_inmodel_obb(pil_image))
        elif model == "comprehensive":
            result.update(await test_comprehensive(pil_image))
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

        return result

    except Exception as e:
        return {
            "model": model,
            "success": False,
            "error": str(e),
            "stack_trace": traceback.format_exc(),
        }


async def test_places365(image: Image.Image) -> Dict[str, Any]:
    """Test Places365 scene classification."""
    try:
        from WebInterface.backend.Adapters.Place365 import Place365Adapter
        import os
        from pathlib import Path

        models_dir = Path(__file__).resolve().parents[2] / "backend" / "Models"
        adapter = Place365Adapter(models_dir=str(models_dir))

        # Use the correct method name - 'classify' not 'predict'
        result = adapter.classify(image)
        if result:
            return {
                "success": True,
                "scene": result.get("scene", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "label": result.get("label", "unknown"),
                "raw_result": result,
            }
        else:
            return {"success": False, "error": "No results from Places365 adapter"}

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stack_trace": traceback.format_exc(),
        }


async def test_rfdetr(image: Image.Image) -> Dict[str, Any]:
    """Test RF-DETR object detection."""
    try:
        from WebInterface.backend.Adapters.RFDETR import RFDETRAdapter

        adapter = RFDETRAdapter()
        results = adapter.predict([image])

        if results and len(results) > 0:
            detections = results[0]
            return {"success": True, "detections": detections, "count": len(detections)}
        else:
            return {"success": True, "detections": [], "count": 0}

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stack_trace": traceback.format_exc(),
        }


async def test_inmodel_classification(image: Image.Image) -> Dict[str, Any]:
    """Test InModel YOLOv11 classification only."""
    try:
        from WebInterface.backend.Adapters.InModel import InModelAdapter
        import os
        from pathlib import Path

        models_dir = Path(__file__).resolve().parents[2] / "backend" / "Models"
        custom_dir = models_dir / "CustomModel"

        # Check multiple possible classification model locations
        possible_classification_paths = [
            custom_dir / "CLS.pt",  # In-house model
            custom_dir / "Classification.pt",
            custom_dir / "WaterClsV1.pt",  # From the example
            custom_dir / "classification.pt",
            models_dir / "ObjectDetection" / "yolo11_cls.pt",
        ]

        classification_weights = None
        for path in possible_classification_paths:
            if path.exists():
                classification_weights = str(path)
                break

        # InModel adapter now uses models_dir and looks for InHouse models automatically
        models_dir = Path(__file__).resolve().parents[2] / "backend" / "Models"

        # Debug: Check if models directory exists
        if not models_dir.exists():
            return {
                "success": False,
                "error": f"Models directory not found: {models_dir}",
                "searched_paths": [str(p) for p in possible_classification_paths],
            }

        adapter = InModelAdapter(models_dir=str(models_dir))

        # Use classification method directly
        classification_result = adapter.classify_water(image)

        # Also get model status for debugging
        model_status = adapter.get_model_status()

        return {
            "success": classification_result.get("success", False),
            "classifications": (
                [classification_result] if classification_result.get("success") else []
            ),
            "count": 1 if classification_result.get("success") else 0,
            "classification_weights_used": model_status.get("classification_path"),
            "searched_paths": [str(p) for p in possible_classification_paths],
            "models_dir": str(models_dir),
            "model_status": model_status,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stack_trace": traceback.format_exc(),
        }


async def test_inmodel_obb(image: Image.Image) -> Dict[str, Any]:
    """Test InModel YOLOv11 OBB detection only."""
    try:
        from WebInterface.backend.Adapters.InModel import InModelAdapter
        import os
        from pathlib import Path

        models_dir = Path(__file__).resolve().parents[2] / "backend" / "Models"

        # Try to find OBB weights
        possible_obb_paths = [
            models_dir / "CustomModel" / "OBB.pt",  # In-house model
            models_dir / "CustomModel" / "OBBv1.pt",
            models_dir / "CustomModel" / "obb_weights.pt",
            models_dir / "ObjectDetection" / "yolo11_obb.pt",
        ]

        obb_weights = None
        for path in possible_obb_paths:
            if path.exists():
                obb_weights = str(path)
                break

        # InModel adapter now uses models_dir and looks for InHouse models automatically
        models_dir = Path(__file__).resolve().parents[2] / "backend" / "Models"

        # Debug: Check if models directory exists
        if not models_dir.exists():
            return {
                "success": False,
                "error": f"Models directory not found: {models_dir}",
                "searched_paths": [str(p) for p in possible_obb_paths],
            }

        adapter = InModelAdapter(models_dir=str(models_dir))

        # Use object detection method directly
        detection_result = adapter.detect_objects(image)

        # Also get model status for debugging
        model_status = adapter.get_model_status()

        return {
            "success": detection_result.get("success", False),
            "obb_detections": detection_result.get("detections", []),
            "count": len(detection_result.get("detections", [])),
            "obb_weights_used": model_status.get("obb_path"),
            "searched_paths": [str(p) for p in possible_obb_paths],
            "models_dir": str(models_dir),
            "model_status": model_status,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stack_trace": traceback.format_exc(),
        }


async def test_comprehensive(image: Image.Image) -> Dict[str, Any]:
    """Test comprehensive detection with annotated image output."""
    try:
        from WebInterface.backend.Adapters.InModel import InModelAdapter
        import os
        from pathlib import Path

        models_dir = Path(__file__).resolve().parents[2] / "backend" / "Models"
        custom_dir = models_dir / "CustomModel"

        # Try to find all model weights
        detection_weights = (
            custom_dir / "ObjectDetection.pt"
            if (custom_dir / "ObjectDetection.pt").exists()
            else None
        )
        # Try to find classification weights with priority for in-house model
        classification_paths = [
            custom_dir / "CLS.pt",  # In-house model
            custom_dir / "Classification.pt",
        ]
        classification_weights = None
        for path in classification_paths:
            if path.exists():
                classification_weights = path
                break

        # Look for OBB weights
        possible_obb_paths = [
            custom_dir / "OBB.pt",  # In-house model
            custom_dir / "OBBv1.pt",
            custom_dir / "obb_weights.pt",
            models_dir / "ObjectDetection" / "yolo11_obb.pt",
        ]
        obb_weights = None
        for path in possible_obb_paths:
            if path.exists():
                obb_weights = str(path)
                break

        # InModel adapter now uses models_dir and looks for InHouse models automatically
        # Debug: Check if models directory exists
        if not models_dir.exists():
            return {
                "success": False,
                "error": f"Models directory not found: {models_dir}",
            }

        adapter = InModelAdapter(models_dir=str(models_dir))

        # Run comprehensive classification (matches the example pattern)
        status_log, result_dict, annotated_image = adapter.classify_image_comprehensive(
            image
        )

        # Convert annotated image to base64
        buffer = io.BytesIO()
        annotated_image.save(buffer, format="JPEG", quality=85)
        annotated_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Also get raw comprehensive results and model status
        raw_results = adapter.predict_comprehensive(image)
        model_status = adapter.get_model_status()

        return {
            "success": True,
            "status_log": status_log,
            "final_result": result_dict,
            "annotated_image": annotated_b64,
            "raw_comprehensive": raw_results,
            "models_loaded": {
                "classification": model_status["classification_loaded"],
                "obb": model_status["obb_loaded"],
            },
            "model_status": model_status,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stack_trace": traceback.format_exc(),
        }
