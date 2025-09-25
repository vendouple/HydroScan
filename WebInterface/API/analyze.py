from __future__ import annotations

import json
import os
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except (
    Exception
):  # pragma: no cover - optional dependency for development without OpenCV
    cv2 = None  # type: ignore
    _HAS_CV2 = False

from WebInterface.backend.Adapters.Place365 import Place365Adapter
from WebInterface.backend.Adapters.RFDETR import RFDETRAdapter
from WebInterface.backend.Adapters.UserInput import (
    UserInputAdapter,
    UserInputAssessment,
)
from WebInterface.backend.Adapters.InModel import InModelAdapter
from WebInterface.backend.Ingestion.indonesia.fetch_data import fetch_nearest_station
from WebInterface.backend.Ingestion.indonesia.normalize import normalize_external
from WebInterface.backend.Processing.aggregator import aggregate
from WebInterface.backend.Processing.filters import generate_variants
from WebInterface.backend.Processing.visual_metrics import compute_visual_metrics
from WebInterface.backend.Scoring.scoring import compute_scores


REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = REPO_ROOT / "history"
MODELS_DIR = REPO_ROOT / "WebInterface" / "backend" / "Models"

MAX_IMAGES = 25
MAX_VIDEOS = 5
MAX_VIDEO_DURATION_SECONDS = 60
MAX_VARIANTS = 6
SCENE_SAMPLE_FRAMES = 3
DETECTOR_THRESHOLD = float(os.environ.get("HYDROSCAN_DETECT_THRESHOLD", "0.4"))
MAX_DEBUG_IMAGES = int(os.environ.get("HYDROSCAN_DEBUG_MAX_IMAGES", "24"))


router = APIRouter()

_place_adapter: Optional[Place365Adapter] = None
_rfdetr_adapter: Optional[RFDETRAdapter] = None
_inmodel_adapter: Optional[InModelAdapter] = None
_user_input_adapter: Optional[UserInputAdapter] = None


def _ensure_history_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_place_adapter() -> Place365Adapter:
    global _place_adapter
    if _place_adapter is None:
        _place_adapter = Place365Adapter(models_dir=str(MODELS_DIR))
    return _place_adapter


def _load_rfdetr_adapter() -> RFDETRAdapter:
    global _rfdetr_adapter
    if _rfdetr_adapter is None:
        checkpoint_env = os.environ.get("RFDETR_CHECKPOINT")
        default_checkpoint = MODELS_DIR / "RFDETR" / "rf_detr_m.pth"
        checkpoint = checkpoint_env or (
            str(default_checkpoint) if default_checkpoint.exists() else None
        )
        model_name = os.environ.get("RFDETR_MODEL_NAME", "rf_detr_m")
        _rfdetr_adapter = RFDETRAdapter(
            checkpoint_path=checkpoint,
            model_name=model_name,
        )
    return _rfdetr_adapter


def _load_inmodel_adapter() -> InModelAdapter:
    global _inmodel_adapter
    if _inmodel_adapter is None:
        weights = os.environ.get(
            "INMODEL_WEIGHTS",
            str(MODELS_DIR / "CustomModel" / "OBB.pt"),
        )
        classifier = os.environ.get(
            "INMODEL_CLASSIFIER",
            str(MODELS_DIR / "CustomModel" / "CLS.pt"),
        )
        _inmodel_adapter = InModelAdapter(
            weights_path=weights, classification_path=classifier
        )
    return _inmodel_adapter


def _load_user_input_adapter() -> UserInputAdapter:
    global _user_input_adapter
    if _user_input_adapter is None:
        models_dir = MODELS_DIR / "UserInput"
        _user_input_adapter = UserInputAdapter(models_dir=str(models_dir))
    return _user_input_adapter


def _prepare_user_input_analysis(
    description: Optional[str],
    scene_majority: str,
    aggregation_result: Dict[str, Any],
    normalized_external: Dict[str, Any],
    metrics_per_frame: List[Dict[str, float]],
    media_info: Dict[str, int],
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Optional[UserInputAssessment],
    Optional[Dict[str, Any]],
]:
    score_context = {
        "visual_avg": aggregation_result.get("metrics_avg", {}),
        "detections": aggregation_result.get("detections", []),
        "metrics_by_frame": metrics_per_frame,
        "external": normalized_external,
        "user_text": description,
        "media_info": media_info,
    }

    baseline_scores = compute_scores(score_context)

    assessment: Optional[UserInputAssessment] = None
    user_analysis: Optional[Dict[str, Any]] = None
    if description:
        adapter = _load_user_input_adapter()
        assessment = adapter.analyze(
            description,
            {
                "detections": aggregation_result.get("detections", []),
                "top_detection": aggregation_result.get("top_detection", {}),
                "visual_metrics": aggregation_result.get("metrics_avg", {}),
                "external": normalized_external,
                "scene": scene_majority,
                "base_scores": baseline_scores,
            },
        )
        user_analysis = assessment.as_dict()
        score_context["user_text_assessment"] = user_analysis

    return score_context, baseline_scores, assessment, user_analysis


def _select_detector_adapter() -> RFDETRAdapter | InModelAdapter:
    backend = os.environ.get("MODEL_BACKEND", "rfdetr").lower()
    adapter = (
        _load_rfdetr_adapter() if backend != "inmodel" else _load_inmodel_adapter()
    )
    if adapter.model is None and backend != "inmodel":
        fallback = _load_inmodel_adapter()
        if fallback.model is not None:
            return fallback
    return adapter


async def _save_upload(file: UploadFile, target_dir: Path) -> Path:
    filename = file.filename or f"upload-{uuid.uuid4()}"
    dest = target_dir / filename
    with dest.open("wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
    await file.seek(0)
    return dest


def _is_video(upload: UploadFile) -> bool:
    if upload.content_type and upload.content_type.startswith("video/"):
        return True
    extension = (upload.filename or "").lower()
    return extension.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _extract_video_duration(cap: Any) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if fps <= 0:
        return 0.0
    return frame_count / fps


def _extract_frames_from_video(
    path: Path, max_frames: int = 45
) -> Sequence[Image.Image]:
    if not _HAS_CV2:
        return []
    frames: List[Image.Image] = []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return frames
    duration = _extract_video_duration(cap)
    if duration > MAX_VIDEO_DURATION_SECONDS:
        cap.release()
        raise ValueError("Video exceeds maximum allowed duration")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames == 0:
        total_frames = int(duration * 24)  # heuristic fallback
    step = max(1, total_frames // max_frames)
    idx = 0
    while cap.isOpened() and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        idx += step
    cap.release()
    return frames


def _majority_vote(labels: Sequence[str]) -> str:
    if not labels:
        return "unknown"
    counter = Counter(labels)
    return counter.most_common(1)[0][0]


def _serialize_for_json(data: Any) -> Any:
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, datetime):
        return data.isoformat()
    if isinstance(data, dict):
        return {k: _serialize_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_serialize_for_json(v) for v in data]
    if isinstance(data, tuple):
        return [_serialize_for_json(v) for v in data]
    return data


def _run_detector(
    adapter: RFDETRAdapter | InModelAdapter, image: Image.Image, threshold: float
) -> List[Dict[str, Any]]:
    try:
        if isinstance(adapter, InModelAdapter):
            results = adapter.predict([image], conf=threshold)
            return results[0] if results else []
        if hasattr(adapter, "predict"):
            return adapter.predict(image, threshold=threshold)  # type: ignore[arg-type]
    except Exception:
        return []
    return []


def _sanitize_slug(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "-", (value or "").strip().lower())
    return clean or "variant"


def _annotate_detections(
    image: Image.Image, detections: List[Dict[str, Any]]
) -> Image.Image:
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover - pillow default font safe fallback
        font = None

    for det in detections:
        bbox = det.get("bbox") or det.get("box")
        if not bbox or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        except Exception:
            continue

        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)

        draw.rectangle([(x1, y1), (x2, y2)], outline=(96, 165, 250), width=3)
        label = det.get("class_name") or det.get("label") or "object"
        score = det.get("score")
        if isinstance(score, (int, float)):
            label = f"{label} {score:.2f}"

        if font:
            text_width, text_height = draw.textsize(label, font=font)
        else:  # pragma: no cover - extremely rare path
            text_width, text_height = len(label) * 6, 10

        text_bg_x1 = x1
        text_bg_y1 = y1 - text_height - 6
        if text_bg_y1 < 0:
            text_bg_y1 = y1 + 4
        text_bg_x2 = text_bg_x1 + text_width + 8
        text_bg_y2 = text_bg_y1 + text_height + 6

        draw.rectangle(
            [(text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2)],
            fill=(15, 23, 42),
        )
        text_pos = (text_bg_x1 + 4, text_bg_y1 + 3)
        draw.text(text_pos, label, fill=(240, 249, 255), font=font)

    return annotated


@router.post("/analyze")
async def analyze_endpoint(
    files: List[UploadFile] = File(..., description="Images and/or videos"),
    description: Optional[str] = Form(
        None, description="User description: smell/color/feel/temperature"
    ),
    lat: Optional[float] = Form(None, description="Latitude for external data"),
    lon: Optional[float] = Form(None, description="Longitude for external data"),
    debug: bool = Form(False),
):
    if not files:
        raise HTTPException(status_code=400, detail="No media uploaded")

    images: List[Path] = []
    videos: List[Path] = []
    if len(files) > MAX_IMAGES + MAX_VIDEOS:
        raise HTTPException(status_code=400, detail="Too many files uploaded")

    for upload in files:
        if _is_video(upload):
            videos.append(Path(upload.filename or "video"))
        else:
            images.append(Path(upload.filename or "image"))

    if len(images) > MAX_IMAGES:
        raise HTTPException(
            status_code=400, detail=f"Maximum {MAX_IMAGES} images allowed"
        )
    if len(videos) > MAX_VIDEOS:
        raise HTTPException(
            status_code=400, detail=f"Maximum {MAX_VIDEOS} videos allowed"
        )

    analysis_id = str(uuid.uuid4())
    target_dir = HISTORY_DIR / analysis_id
    _ensure_history_dir(target_dir)

    saved_files: List[Dict[str, Any]] = []
    actual_image_paths: List[Path] = []
    actual_video_paths: List[Path] = []
    debug_images_manifest: List[Dict[str, Any]] = []
    debug_dir: Optional[Path] = None
    debug_detections_dir: Optional[Path] = None

    if debug:
        debug_dir = target_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_detections_dir = debug_dir / "detections"
        debug_detections_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        dest = await _save_upload(upload, target_dir)
        record = {
            "filename": upload.filename,
            "content_type": upload.content_type,
            "path": str(dest),
        }
        saved_files.append(record)
        if _is_video(upload):
            actual_video_paths.append(dest)
        else:
            actual_image_paths.append(dest)
        await upload.close()

    frames: List[Image.Image] = []
    metrics_per_frame: List[Dict[str, float]] = []
    timeline: List[Dict[str, Any]] = []
    detector = _select_detector_adapter()
    place365 = _load_place_adapter()

    timeline.append(
        {
            "step": "preparing_media",
            "status": "in-progress",
            "detail": f"{len(actual_image_paths)} images, {len(actual_video_paths)} videos",
        }
    )

    for img_path in actual_image_paths:
        frames.append(_load_image(img_path))

    for video_path in actual_video_paths:
        try:
            video_frames = _extract_frames_from_video(video_path)
            frames.extend(video_frames)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    if not frames:
        raise HTTPException(
            status_code=400, detail="Unable to decode any frames from uploads"
        )

    timeline.append(
        {
            "step": "preparing_media",
            "status": "done",
            "detail": f"Total frames for analysis: {len(frames)}",
        }
    )

    sampled_scene_frames = frames[:SCENE_SAMPLE_FRAMES]
    scene_results = [place365.classify(frame) for frame in sampled_scene_frames]
    scene_labels = [res.get("scene", "unknown") for res in scene_results]
    scene_majority = _majority_vote(scene_labels)
    mean_scene_conf = (
        sum(res.get("confidence", 0.0) for res in scene_results) / len(scene_results)
        if scene_results
        else 0.0
    )
    timeline.append(
        {
            "step": "scene_classifier",
            "status": "done",
            "detail": f"scene={scene_majority} ({mean_scene_conf:.2f})",
        }
    )

    frames_for_aggregation: List[Dict[str, Any]] = []
    total_variants = 0

    timeline.append({"step": "applying_filters", "status": "in-progress"})
    for idx, frame in enumerate(frames):
        variants = generate_variants(frame, max_variants=MAX_VARIANTS)
        total_variants += len(variants)
        original_metrics = compute_visual_metrics(frame)
        metrics_per_frame.append(original_metrics)
        for variant_name, variant_img in variants:
            metrics = compute_visual_metrics(variant_img)
            detections = _run_detector(detector, variant_img, DETECTOR_THRESHOLD)
            if not detections and detector.model is None:
                timeline.append(
                    {
                        "step": "running_detector",
                        "status": "warning",
                        "detail": "Detector model unavailable",
                    }
                )

            debug_image_rel_path: Optional[str] = None
            if (
                debug
                and detections
                and debug_detections_dir is not None
                and len(debug_images_manifest) < MAX_DEBUG_IMAGES
            ):
                annotated = _annotate_detections(variant_img, detections)
                variant_slug = _sanitize_slug(variant_name)
                filename = f"frame{idx:03d}_{variant_slug}.jpg"
                annotated_path = debug_detections_dir / filename
                try:
                    annotated.save(annotated_path, format="JPEG", quality=90)
                    debug_image_rel_path = f"detections/{filename}"
                    debug_images_manifest.append(
                        {
                            "frame_index": idx,
                            "variant": variant_name,
                            "detections": len(detections),
                            "relative_path": debug_image_rel_path,
                            "filename": filename,
                        }
                    )
                except Exception:
                    pass

            frames_for_aggregation.append(
                {
                    "frame_index": idx,
                    "variant": variant_name,
                    "metrics": metrics,
                    "detections": detections,
                    "debug_image": debug_image_rel_path,
                }
            )

    aggregation_result = aggregate(frames_for_aggregation)
    timeline.append(
        {
            "step": "applying_filters",
            "status": "done",
            "detail": f"variants={total_variants}",
        }
    )

    aggregated_detections = aggregation_result.get("detections", [])
    top_detection = aggregation_result.get("top_detection")
    detection_detail = f"Detections={len(aggregated_detections)}" + (
        f", top={top_detection.get('class_name')} ({top_detection.get('score', 0.0):.2f})"
        if top_detection
        else ""
    )
    timeline.append(
        {
            "step": "running_detector",
            "status": "done" if aggregated_detections else "warning",
            "detail": detection_detail,
        }
    )
    timeline.append(
        {
            "step": "aggregating_results",
            "status": "done",
            "detail": f"metrics={len(aggregation_result.get('metrics_avg', {}))}",
        }
    )
    water_detection_message = (
        "Water wasn't detected in the provided media. "
        "Please upload clearer samples or ensure water regions are visible."
    )

    media_info = {
        "media_count": len(actual_image_paths) + len(actual_video_paths),
        "variant_count": total_variants,
        "frame_count": len(frames),
    }

    if not aggregated_detections:
        timeline.append(
            {
                "step": "running_detector",
                "status": "error",
                "detail": water_detection_message,
            }
        )

        timeline.append(
            {
                "step": "finalizing",
                "status": "in-progress",
                "detail": "Preparing no-detection response",
            }
        )

        user_analysis: Optional[Dict[str, Any]] = None
        assessment: Optional[UserInputAssessment] = None
        if description:
            timeline.append(
                {
                    "step": "user_input_analysis",
                    "status": "in-progress",
                    "detail": "Assessing user notes with LLaMA 3",
                }
            )
            _, _, assessment, user_analysis = _prepare_user_input_analysis(
                description,
                scene_majority,
                aggregation_result,
                {},
                metrics_per_frame,
                media_info,
            )
            if assessment and assessment.available:
                timeline.append(
                    {
                        "step": "user_input_analysis",
                        "status": "done",
                        "detail": f"LLM confidence {assessment.confidence:.0f}%",
                    }
                )
            else:
                timeline.append(
                    {
                        "step": "user_input_analysis",
                        "status": "warning",
                        "detail": (
                            assessment.reason
                            if assessment and assessment.reason
                            else "User input model unavailable"
                        ),
                    }
                )

        result_payload = {
            "analysis_id": analysis_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "description": description,
            "scene": {
                "majority": scene_majority,
                "samples": scene_results,
                "confidence": mean_scene_conf,
            },
            "media": {
                "saved_files": saved_files,
                "frame_count": len(frames),
                "variant_count": total_variants,
            },
            "aggregation": aggregation_result,
            "timeline": timeline,
            "status": "no_water_detected",
            "message": water_detection_message,
            "user_analysis": user_analysis,
            "debug": {
                "enabled": debug,
                "detection_images": debug_images_manifest if debug else [],
            },
        }

        result_path = target_dir / "result.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(
                _serialize_for_json(result_payload), f, ensure_ascii=False, indent=2
            )

        response: Dict[str, Any] = {
            "analysis_id": analysis_id,
            "status": "no_water_detected",
            "message": water_detection_message,
            "result_path": str(result_path),
            "timeline": timeline,
            "aggregation": aggregation_result,
        }
        if user_analysis is not None:
            response["user_analysis"] = user_analysis

        if debug:
            debug_path = target_dir / "debug.json"
            with debug_path.open("w", encoding="utf-8") as f:
                json.dump(
                    _serialize_for_json(
                        {
                            "frames": frames_for_aggregation,
                            "metrics": metrics_per_frame,
                            "detection_images": debug_images_manifest,
                        }
                    ),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            response["debug_path"] = str(debug_path)
            base_url = f"/api/results/{analysis_id}/artifacts"
            response["debug_images"] = [
                {
                    **img,
                    "url": "{}/{}".format(
                        base_url,
                        str(img.get("relative_path", "")).replace("\\", "/"),
                    ),
                }
                for img in debug_images_manifest
            ]

        timeline.append(
            {
                "step": "finalizing",
                "status": "done",
                "detail": "Results saved without detections",
            }
        )
        response["timeline"] = timeline

        return response

    if debug:
        if debug_images_manifest:
            timeline.append(
                {
                    "step": "debug_detections",
                    "status": "done",
                    "detail": f"Captured {len(debug_images_manifest)} annotated frames",
                }
            )
        else:
            timeline.append(
                {
                    "step": "debug_detections",
                    "status": "warning",
                    "detail": "No detections captured for debug",
                }
            )

    external_raw: Dict[str, Any] | None = None
    normalized_external: Dict[str, Any] = {}
    if lat is not None and lon is not None:
        timeline.append({"step": "external_data", "status": "in-progress"})
        try:
            external_raw = fetch_nearest_station(lat=lat, lon=lon)
            normalized_external = normalize_external(external_raw)
            if normalized_external:
                distance = normalized_external.get("distance_km")
                if isinstance(distance, (int, float)):
                    distance_detail = f"{distance:.2f} km"
                else:
                    distance_detail = "unknown distance"
                timeline.append(
                    {
                        "step": "external_data",
                        "status": "done",
                        "detail": f"station={normalized_external.get('station_id')} distance={distance_detail}",
                    }
                )
            else:
                timeline.append(
                    {
                        "step": "external_data",
                        "status": "warning",
                        "detail": "No station found within range",
                    }
                )
        except Exception as exc:
            timeline.append(
                {"step": "external_data", "status": "error", "detail": str(exc)}
            )
            normalized_external = {}

    if description:
        timeline.append(
            {
                "step": "user_input_analysis",
                "status": "in-progress",
                "detail": "Assessing user notes with LLaMA 3",
            }
        )

    score_context, _baseline_scores, assessment, user_analysis = (
        _prepare_user_input_analysis(
            description,
            scene_majority,
            aggregation_result,
            normalized_external,
            metrics_per_frame,
            media_info,
        )
    )

    if description:
        if assessment and assessment.available:
            timeline.append(
                {
                    "step": "user_input_analysis",
                    "status": "done",
                    "detail": f"LLM confidence {assessment.confidence:.0f}%",
                }
            )
        else:
            timeline.append(
                {
                    "step": "user_input_analysis",
                    "status": "warning",
                    "detail": (
                        assessment.reason
                        if assessment and assessment.reason
                        else "User input model unavailable"
                    ),
                }
            )

    scores = compute_scores(score_context)
    timeline.append(
        {"step": "scoring", "status": "done", "detail": scores["band_label"]}
    )

    result_payload = {
        "analysis_id": analysis_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "description": description,
        "scene": {
            "majority": scene_majority,
            "samples": scene_results,
            "confidence": mean_scene_conf,
        },
        "external_data": normalized_external,
        "media": {
            "saved_files": saved_files,
            "frame_count": len(frames),
            "variant_count": total_variants,
        },
        "aggregation": aggregation_result,
        "scores": scores,
        "timeline": timeline,
        "user_analysis": user_analysis,
        "debug": {
            "enabled": debug,
            "detection_images": debug_images_manifest if debug else [],
        },
    }

    timeline.append(
        {
            "step": "finalizing",
            "status": "done",
            "detail": "Results saved",
        }
    )

    result_path = target_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(_serialize_for_json(result_payload), f, ensure_ascii=False, indent=2)

    response = {
        "analysis_id": analysis_id,
        "status": "completed",
        "result_path": str(result_path),
    }
    if debug:
        debug_path = target_dir / "debug.json"
        with debug_path.open("w", encoding="utf-8") as f:
            json.dump(
                _serialize_for_json(
                    {
                        "frames": frames_for_aggregation,
                        "metrics": metrics_per_frame,
                        "detection_images": debug_images_manifest,
                    }
                ),
                f,
                ensure_ascii=False,
                indent=2,
            )
        response["debug_path"] = str(debug_path)
        base_url = f"/api/results/{analysis_id}/artifacts"
        response["debug_images"] = [
            {
                **img,
                "url": "{}/{}".format(
                    base_url,
                    str(img.get("relative_path", "")).replace("\\", "/"),
                ),
            }
            for img in debug_images_manifest
        ]

    response["scores"] = scores
    response["scene"] = result_payload["scene"]
    if normalized_external:
        response["external_data"] = normalized_external
    response["timeline"] = timeline
    response["media"] = result_payload["media"]
    response["aggregation"] = aggregation_result
    if user_analysis is not None:
        response["user_analysis"] = user_analysis

    return response
