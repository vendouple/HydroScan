from fastapi import APIRouter, UploadFile, File, Form
from typing import List, Optional
import uuid
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HISTORY_DIR = os.path.join(REPO_ROOT, "history")


router = APIRouter()


@router.post("/analyze")
async def analyze_endpoint(
    files: List[UploadFile] = File(..., description="Images and/or videos"),
    description: Optional[str] = Form(
        None, description="User description: smell/color/feel/temperature"
    ),
    debug: bool = Form(False),
):
    # For now, just return a generated analysis_id and echo counts
    analysis_id = str(uuid.uuid4())
    target_dir = os.path.join(HISTORY_DIR, analysis_id)
    os.makedirs(target_dir, exist_ok=True)

    saved = []
    for f in files:
        dest = os.path.join(target_dir, f.filename)
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved.append(dest)

    return {
        "analysis_id": analysis_id,
        "received_files": len(files),
        "saved": saved,
        "debug": debug,
        "message": "Uploaded files saved; processing pipeline pending",
    }
