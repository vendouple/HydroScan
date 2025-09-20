from __future__ import annotations
import os
import json
from typing import Any
import numpy as np
from loguru import logger
from ..config import settings
from ..utils.media import load_image, sample_video_frames, apply_filters
from ..ml.scene_places365 import Places365SceneClassifier
from ..ml.yolo11_adapter import YOLO11Adapter
from ..ml.detector_base import Detection, DetectorAdapter
from .store import Store
import pytesseract
import httpx
from ..utils.cache import FileTTLCache


class AnalysisService:
    def __init__(self, store: Store):
        self.store = store
        # detector selection (default yolo11 per user)
        backend = settings.detector_backend
        model_path = settings.detector_model
        device = settings.detector_device
        if backend == "yolo11":
            self.detector: DetectorAdapter = YOLO11Adapter(model_path, device)
        elif backend == "yolo12":
            from ..ml.yolo12_adapter import YOLO12Adapter  # raises on init

            self.detector = YOLO12Adapter(model_path, device)
        else:
            raise RuntimeError(
                "Currently only yolo11 backend is implemented in this build"
            )
        self.scene_classifier = Places365SceneClassifier(
            cache_dir=settings.cache_dir,
            skip_download=settings.ml_skip_download,
            device=device,
        )
        self.cache = FileTTLCache(settings.cache_dir, settings.cache_ttl_s)

    async def process_job(self, job_id: str):
        job = await self.store.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        timeline: list[dict[str, Any]] = []
        images: list[np.ndarray] = []

        # Load media
        for path in job["media"]:
            ext = os.path.splitext(path)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                try:
                    images.append(load_image(path))
                except Exception as e:
                    timeline.append(
                        {"step": "load_image", "path": path, "error": str(e)}
                    )
            elif ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
                try:
                    frames = sample_video_frames(path)
                    timeline.append(
                        {"step": "sample_video", "path": path, "frames": len(frames)}
                    )
                    images.extend(frames)
                except Exception as e:
                    timeline.append(
                        {"step": "sample_video", "path": path, "error": str(e)}
                    )

        if not images:
            await self.store.set_result(
                job_id,
                {
                    "id": job_id,
                    "potability": 0.0,
                    "confidence": 0.0,
                    "scene": "Unknown",
                    "timeline": timeline + [{"error": "no media"}],
                },
            )
            return

        # Scene classification on first image
        scene_res = self.scene_classifier.classify(images[0])
        timeline.append({"step": "scene", **scene_res})

        # Apply filters on first image only for quick comparison
        variants = apply_filters(images[0])
        variant_scores = []

        # Run detector on first few images for speed
        sample_for_det = images[: min(8, len(images))]
        dets = self.detector.predict(sample_for_det)
        det_count = sum(len(d) for d in dets)
        timeline.append({"step": "detector", "count": det_count})

        # OCR brand if any detection likely bottle/label
        brand_text = ""
        if det_count > 0:
            # naive crop top detection
            import cv2

            img = sample_for_det[0]
            h, w, _ = img.shape
            for d in dets[0]:
                x1, y1, x2, y2 = [int(v) for v in d.bbox_xyxy]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)
                crop = img[y1:y2, x1:x2]
                try:
                    txt = pytesseract.image_to_string(crop)
                    brand_text += " " + txt.strip()
                except Exception:
                    pass
            brand_text = brand_text.strip()
            timeline.append({"step": "ocr", "text": brand_text[:120]})

        # External context (US only for now)
        loc_context = {}
        try:
            # This is a placeholder call site; actual lat/lon to be provided by UI later
            pass
        except Exception as e:
            timeline.append({"step": "context", "error": str(e)})

        # Potability heuristic
        pot = 0.5  # start neutral
        conf = 0.5
        if scene_res["scene"].startswith("Outdoor"):
            pot += 0.05
        if det_count == 0:
            pot -= 0.1
        if brand_text:
            conf += 0.1
        # User inputs signal
        if job.get("user_inputs"):
            txt = job["user_inputs"].lower()
            if any(
                w in txt
                for w in ["smell", "odor", "odour", "sulfur", "chlorine", "rust"]
            ):
                pot -= 0.05
            if any(w in txt for w in ["clear", "no smell", "transparent"]):
                pot += 0.03
        pot = float(max(0.0, min(1.0, pot)))
        conf = float(max(0.0, min(1.0, conf)))

        result = {
            "id": job_id,
            "potability": pot * 100.0,
            "confidence": conf * 100.0,
            "scene": scene_res["scene"],
            "timeline": timeline,
        }
        await self.store.set_result(job_id, result)

    async def brand_lookup(self, q: str) -> dict[str, Any]:
        # Wikipedia search (no key required)
        out = {"query": q, "results": []}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "format": "json",
                        "srsearch": q,
                    },
                )
                if r.status_code == 200:
                    js = r.json()
                    for item in js.get("query", {}).get("search", [])[:5]:
                        out["results"].append(
                            {
                                "title": item.get("title"),
                                "snippet": item.get("snippet"),
                                "pageid": item.get("pageid"),
                                "url": f"https://en.wikipedia.org/?curid={item.get('pageid')}",
                            }
                        )
        except Exception as e:
            out["error"] = str(e)
        return out

    async def location_context(self, lat: float, lon: float) -> dict[str, Any]:
        key = f"usgs_wqp_{lat:.3f}_{lon:.3f}"
        cached = self.cache.get(key)
        if cached:
            return cached
        out = {"lat": lat, "lon": lon, "sources": []}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # Water Quality Portal: stations within radius
                r = await client.get(
                    "https://www.waterqualitydata.us/Station/search",
                    params={
                        "within": int(settings.location_radius_km),
                        "lat": lat,
                        "long": lon,
                        "mimeType": "json",
                    },
                )
                if r.status_code == 200:
                    out["sources"].append({"name": "WQP Stations", "data": r.json()})
        except Exception as e:
            out.setdefault("errors", []).append(str(e))
        self.cache.set(key, out)
        return out
