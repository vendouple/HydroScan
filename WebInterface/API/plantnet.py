"""PlantNet plant identification API endpoints."""
from __future__ import annotations

import base64
import traceback
from io import BytesIO
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from WebInterface.backend.Adapters.PlantNet import PlantNetAdapter, PlantNetError

router = APIRouter(prefix="/plantnet", tags=["plantnet"])

# Initialize adapter
_adapter: Optional[PlantNetAdapter] = None


def get_adapter() -> PlantNetAdapter:
    """Get or create PlantNet adapter singleton."""
    global _adapter
    if _adapter is None:
        _adapter = PlantNetAdapter()
    return _adapter


@router.post("/identify")
async def identify_plant(
    file: UploadFile = File(...),
    organs: str = Form(default="auto"),
):
    """
    Identify plant from uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        organs: Comma-separated list of plant organs visible in image
                (flower, leaf, fruit, bark, auto). Default: auto
    
    Returns:
        Plant identification results with scientific names, common names,
        confidence scores, and reference images.
    """
    try:
        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(BytesIO(contents))
            image.verify()
            # Reopen after verify
            image = Image.open(BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        
        # Parse organs
        organ_list = [o.strip().lower() for o in organs.split(",") if o.strip()]
        if not organ_list:
            organ_list = ["auto"]
        
        # Validate organs
        valid_organs = {"flower", "leaf", "fruit", "bark", "auto"}
        for organ in organ_list:
            if organ not in valid_organs:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid organ '{organ}'. Must be one of: {', '.join(valid_organs)}"
                )
        
        # Call PlantNet API
        adapter = get_adapter()
        result = adapter.identify_from_pil(image, organs=organ_list)
        
        # Format response
        plants = []
        for r in result.results[:10]:  # Top 10 results
            plants.append({
                "scientific_name": r.scientific_name,
                "common_names": r.common_names[:5] if r.common_names else [],
                "family": r.family,
                "genus": r.genus,
                "confidence": round(r.score * 100, 2),
                "reference_images": [
                    img.get("url", {}).get("m", "") 
                    for img in r.images[:3]
                ]
            })
        
        return {
            "success": True,
            "best_match": result.best_match,
            "plants": plants,
            "remaining_requests": result.remaining_requests,
            "query_organs": organ_list,
        }
    
    except PlantNetError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Identification failed: {e}")


@router.post("/identify-multiple")
async def identify_plants_multiple(
    files: List[UploadFile] = File(...),
    organs: str = Form(default="auto"),
):
    """
    Identify plants from multiple uploaded images.
    
    Args:
        files: List of image files (max 5)
        organs: Comma-separated list of plant organs visible in images
    
    Returns:
        List of identification results for each image.
    """
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 images allowed")
    
    results = []
    for i, file in enumerate(files):
        try:
            # Read image
            contents = await file.read()
            try:
                image = Image.open(BytesIO(contents))
                image.verify()
                image = Image.open(BytesIO(contents))
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": f"Invalid image: {e}"
                })
                continue
            
            # Parse organs
            organ_list = [o.strip().lower() for o in organs.split(",") if o.strip()]
            if not organ_list:
                organ_list = ["auto"]
            
            # Call API
            adapter = get_adapter()
            result = adapter.identify_from_pil(image, organs=organ_list)
            
            plants = []
            for r in result.results[:5]:  # Top 5 per image
                plants.append({
                    "scientific_name": r.scientific_name,
                    "common_names": r.common_names[:3] if r.common_names else [],
                    "family": r.family,
                    "confidence": round(r.score * 100, 2),
                })
            
            results.append({
                "filename": file.filename,
                "success": True,
                "best_match": result.best_match,
                "plants": plants,
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "results": results,
        "total_images": len(files),
    }


@router.get("/health")
async def plantnet_health():
    """Check PlantNet adapter health."""
    try:
        adapter = get_adapter()
        return {
            "status": "ok",
            "api_configured": bool(adapter.api_key),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
