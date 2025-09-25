from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query


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

    return result
