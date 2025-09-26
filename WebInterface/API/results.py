from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
import os

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

    debug_section = result.get("debug") or {}
    snapshots = debug_section.get("snapshots") or {}

    if include_debug:
        debug_path = analysis_dir / "debug.json"
        if debug_path.exists():
            with debug_path.open("r", encoding="utf-8") as f:
                result["debug_artifacts"] = json.load(f)
        else:
            result["debug_artifacts"] = None

    if snapshots:
        base_url = f"/api/results/{analysis_id}/artifacts"
        for category, entries in snapshots.items():
            for entry in entries:
                rel = entry.get("relative_path") or entry.get("filename")
                if rel:
                    entry["url"] = "{}/{}".format(base_url, str(rel).replace("\\", "/"))

    legacy_manifest = debug_section.get("detection_images") or []
    if legacy_manifest:
        base_url = f"/api/results/{analysis_id}/artifacts"
        for entry in legacy_manifest:
            rel = entry.get("relative_path") or entry.get("filename")
            if rel:
                entry["url"] = "{}/{}".format(base_url, str(rel).replace("\\", "/"))

    result["debug"] = debug_section
    if snapshots:
        result["debug"]["snapshots"] = snapshots
    if legacy_manifest:
        result["debug"]["detection_images"] = legacy_manifest

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


@router.get("/history")
async def list_analysis_history(limit: int = Query(50, ge=1, le=200)) -> Dict[str, Any]:
    """List recent analysis history with metadata."""
    if not HISTORY_DIR.exists():
        return {"analyses": [], "total": 0}

    analyses = []

    # Get all UUID directories in history
    for analysis_dir in HISTORY_DIR.iterdir():
        if not analysis_dir.is_dir() or analysis_dir.name.startswith("."):
            continue

        result_path = analysis_dir / "result.json"
        if not result_path.exists():
            continue

        try:
            with result_path.open("r", encoding="utf-8") as f:
                result = json.load(f)

            # Extract key metadata
            analysis_id = result.get("analysis_id", analysis_dir.name)
            timestamp_str = result.get("timestamp")

            # Parse timestamp for sorting
            timestamp = None
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                except Exception:
                    timestamp = datetime.fromtimestamp(result_path.stat().st_mtime)
            else:
                timestamp = datetime.fromtimestamp(result_path.stat().st_mtime)

            # Get potability and confidence scores
            potability_score = result.get("potability_score", 0)
            confidence_score = result.get("confidence_score", 0)
            band_label = result.get("band_label", "Unknown")

            # Check if debug data is available
            debug_available = (analysis_dir / "debug.json").exists()

            # Get scene information
            scene_info = result.get("scene", {})
            scene_label = scene_info.get("majority", "unknown")

            # Count media files (excluding debug artifacts)
            media_files = []
            for ext in [".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"]:
                media_files.extend(analysis_dir.glob(f"*{ext}"))
                media_files.extend(analysis_dir.glob(f"*{ext.upper()}"))

            analyses.append(
                {
                    "analysis_id": analysis_id,
                    "timestamp": timestamp_str,
                    "timestamp_parsed": timestamp.isoformat(),
                    "potability_score": potability_score,
                    "confidence_score": confidence_score,
                    "band_label": band_label,
                    "scene": scene_label,
                    "debug_available": debug_available,
                    "media_count": len(media_files),
                    "description": result.get("description"),
                }
            )

        except Exception as e:
            # Skip analyses that can't be parsed
            print(f"[HydroScan] Error parsing analysis {analysis_dir.name}: {e}")
            continue

    # Sort by timestamp (newest first)
    analyses.sort(key=lambda x: x["timestamp_parsed"], reverse=True)

    # Apply limit
    limited_analyses = analyses[:limit]

    return {
        "analyses": limited_analyses,
        "total": len(analyses),
        "showing": len(limited_analyses),
    }


@router.delete("/history/{analysis_id}")
async def delete_analysis(analysis_id: str) -> Dict[str, Any]:
    """Delete a specific analysis from history."""
    analysis_dir = HISTORY_DIR / analysis_id

    if not analysis_dir.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        # Remove the entire analysis directory
        import shutil

        shutil.rmtree(analysis_dir)

        return {
            "success": True,
            "message": f"Analysis {analysis_id} deleted successfully",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete analysis: {str(e)}"
        )


@router.post("/history/clear")
async def clear_history(keep_recent: int = Query(0, ge=0, le=100)) -> Dict[str, Any]:
    """Clear analysis history, optionally keeping the most recent N analyses."""
    if not HISTORY_DIR.exists():
        return {"success": True, "deleted": 0, "kept": 0}

    try:
        # Get all analysis directories
        analyses = []
        for analysis_dir in HISTORY_DIR.iterdir():
            if not analysis_dir.is_dir() or analysis_dir.name.startswith("."):
                continue

            result_path = analysis_dir / "result.json"
            if not result_path.exists():
                continue

            # Get timestamp for sorting
            try:
                with result_path.open("r", encoding="utf-8") as f:
                    result = json.load(f)
                timestamp_str = result.get("timestamp")
                if timestamp_str:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                else:
                    timestamp = datetime.fromtimestamp(result_path.stat().st_mtime)
            except Exception:
                timestamp = datetime.fromtimestamp(result_path.stat().st_mtime)

            analyses.append((analysis_dir, timestamp))

        # Sort by timestamp (newest first)
        analyses.sort(key=lambda x: x[1], reverse=True)

        # Keep the most recent N analyses
        to_keep = analyses[:keep_recent]
        to_delete = analyses[keep_recent:]

        # Delete the rest
        deleted_count = 0
        import shutil

        for analysis_dir, _ in to_delete:
            try:
                shutil.rmtree(analysis_dir)
                deleted_count += 1
            except Exception as e:
                print(f"[HydroScan] Failed to delete {analysis_dir.name}: {e}")

        return {"success": True, "deleted": deleted_count, "kept": len(to_keep)}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear history: {str(e)}"
        )
