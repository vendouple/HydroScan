from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import shutil
from .config import settings
import json as _json
from fastapi.openapi.utils import get_openapi
from .services.processor import AnalysisService
from .services.store import Store

app = FastAPI(title="HydroScan API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = Store(db_url=settings.database_url)
processor = AnalysisService(store=store)


@app.on_event("startup")
async def _dump_openapi():
    os.makedirs("contracts", exist_ok=True)
    spec = get_openapi(title=app.title, version=app.version, routes=app.routes)
    with open("contracts/openapi.json", "w", encoding="utf-8") as f:
        _json.dump(spec, f, indent=2)


class AnalyzeResponse(BaseModel):
    id: str
    status: str


class AnalyzeResult(BaseModel):
    id: str
    potability: float
    confidence: float
    scene: str
    timeline: list


@app.get("/status")
async def status():
    return {
        "ok": True,
        "model": processor.detector.model_info(),
        "scene": processor.scene_classifier.info(),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    background_tasks: BackgroundTasks,
    images: Optional[List[UploadFile]] = File(None),
    videos: Optional[List[UploadFile]] = File(None),
    user_inputs: Optional[str] = Form(None),
    debug: Optional[str] = Form("none"),
):
    if not images and not videos:
        raise HTTPException(
            status_code=400, detail="Provide at least one image or video"
        )

    os.makedirs(settings.media_dir, exist_ok=True)
    media_paths: list[str] = []

    if images:
        if len(images) > 25:
            raise HTTPException(status_code=400, detail="Max 25 images")
        for f in images:
            ext = os.path.splitext(f.filename)[1].lower()
            uid = f"img_{uuid.uuid4().hex}{ext}"
            path = os.path.join(settings.media_dir, uid)
            with open(path, "wb") as out:
                shutil.copyfileobj(f.file, out)
            media_paths.append(path)

    if videos:
        if len(videos) > 5:
            raise HTTPException(status_code=400, detail="Max 5 videos")
        for f in videos:
            ext = os.path.splitext(f.filename)[1].lower()
            uid = f"vid_{uuid.uuid4().hex}{ext}"
            path = os.path.join(settings.media_dir, uid)
            with open(path, "wb") as out:
                shutil.copyfileobj(f.file, out)
            media_paths.append(path)

    job_id = uuid.uuid4().hex
    await store.create_job(job_id, media_paths, user_inputs or "", debug)

    background_tasks.add_task(processor.process_job, job_id)

    return AnalyzeResponse(id=job_id, status="queued")


@app.get("/results/{job_id}", response_model=AnalyzeResult)
async def results(job_id: str):
    result = await store.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Not found or processing")
    return result


@app.get("/brands/lookup")
async def brand_lookup(q: str):
    data = await processor.brand_lookup(q)
    return JSONResponse(data)


@app.get("/context/location")
async def context_location(lat: float, lon: float):
    data = await processor.location_context(lat, lon)
    return JSONResponse(data)
