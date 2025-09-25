from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse


router = APIRouter()

HISTORY_DIR = Path(__file__).resolve().parents[2] / "history"


@router.get("/results/{analysis_id}")
async def get_results(analysis_id: str, include_debug: bool = Query(False)) -> Any:
    analysis_dir = HISTORY_DIR / analysis_id
    result_path = analysis_dir / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Analysis ID not found")

    with result_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    if include_debug:
        debug_path = analysis_dir / "debug.json"
        if debug_path.exists():
            with debug_path.open("r", encoding="utf-8") as f:
                result["debug_artifacts"] = json.load(f)
        else:
            result["debug_artifacts"] = None

        debug_manifest = result.get("debug", {}).get("detection_images") or []
        base_url = f"/api/results/{analysis_id}/artifacts"
        for entry in debug_manifest:
            rel = entry.get("relative_path") or entry.get("filename")
            if rel:
                entry["url"] = "{}/{}".format(base_url, str(rel).replace("\\", "/"))

    return result


@router.get("/results/{analysis_id}/artifacts/{artifact_path:path}")
async def get_debug_artifact(analysis_id: str, artifact_path: str) -> Any:
    analysis_dir = HISTORY_DIR / analysis_id
    debug_dir = (analysis_dir / "debug").resolve()
    target_path = (debug_dir / artifact_path).resolve()

    if not str(target_path).startswith(str(debug_dir)):
        raise HTTPException(status_code=403, detail="Invalid artifact path")
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(target_path)
