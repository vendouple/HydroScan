from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import numpy as np
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
from WebInterface.backend.Processing.filters import (
    generate_variants,
    get_filter_display_name,
)
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
MAX_FRAMES_PER_VIDEO = int(os.environ.get("HYDROSCAN_MAX_FRAMES_PER_VIDEO", "250"))
FRAME_DIFF_THRESHOLD = float(os.environ.get("HYDROSCAN_FRAME_DIFF_THRESHOLD", "18.0"))
TOP_DETECTION_TARGET = float(os.environ.get("HYDROSCAN_TOP_DETECTION_TARGET", "0.6"))
MIN_DETECTIONS_REQUIRED = int(os.environ.get("HYDROSCAN_MIN_DETECTIONS", "1"))

DEBUG_SNAPSHOT_CATEGORIES = ("scene", "filters", "detector", "ocr")
KNOWN_WATER_BRANDS = {
    "aqua",
    "ades",
    "cleo",
    "club",
    "dasani",
    "evian",
    "nestle",
    "voss",
    "volvic",
}
PACKAGING_KEYWORDS = [
    "bottle",
    "bottled",
    "pack",
    "package",
    "packaging",
    "can",
    "cup",
    "label",
    "logo",
    "brand",
]
WATER_KEYWORDS = [
    "water",
    "liquid",
    "drink",
    "glass",
    "cup",
    "river",
    "lake",
    "stream",
]


def _summarize_classifications(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "samples": [],
        "available": False,
        "dirty_confidence": 0.0,
        "clean_confidence": 0.0,
        "dominant_label": None,
    }

    for pred in predictions:
        if pred.get("detection_type") != "classification":
            continue
        label = (
            pred.get("classification")
            or pred.get("class")
            or pred.get("class_name")
            or ""
        )
        confidence = float(pred.get("confidence") or pred.get("score") or 0.0)
        summary["samples"].append(
            {
                "label": label,
                "confidence": confidence,
                "frame_index": pred.get("frame_index"),
                "source": pred.get("source"),
            }
        )
        summary["available"] = True
        lowered = str(label).lower()
        if "kotor" in lowered or "dirty" in lowered:
            summary["dirty_confidence"] = max(summary["dirty_confidence"], confidence)
        elif "bersih" in lowered or "clean" in lowered:
            summary["clean_confidence"] = max(summary["clean_confidence"], confidence)

    if summary["available"]:
        if summary["dirty_confidence"] >= summary["clean_confidence"]:
            summary["dominant_label"] = "kotor"
        else:
            summary["dominant_label"] = "bersih"

        summary["classification_confidence"] = (
            max(summary["dirty_confidence"], summary["clean_confidence"]) * 100.0
        )
        if summary["dominant_label"] == "kotor":
            summary["classification_score"] = max(
                0.0, (1.0 - summary["dirty_confidence"]) * 40.0
            )
        else:
            summary["classification_score"] = min(
                100.0, 60.0 + summary["clean_confidence"] * 40.0
            )
    else:
        summary["classification_confidence"] = 0.0
        summary["classification_score"] = 50.0

    return summary


def _contains_water_keyword(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return any(keyword in lowered for keyword in WATER_KEYWORDS)


def _append_timeline(
    timeline: List[Dict[str, Any]],
    step: str,
    status: str,
    detail: str,
    progress: Optional[int] = None,
    total_steps: Optional[int] = None,
) -> None:
    entry = {
        "step": step,
        "status": status,
        "detail": detail,
        "timestamp": datetime.now().isoformat(),
    }
    if progress is not None:
        entry["progress"] = progress
    if total_steps is not None:
        entry["total_steps"] = total_steps
        entry["progress_percentage"] = (
            int((progress / total_steps) * 100) if progress is not None else 0
        )
    timeline.append(entry)


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
        default_checkpoint = MODELS_DIR / "RFDETR" / "rf-detr-medium.pth"
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
        # InModel adapter now uses models_dir and looks for CustomModel automatically
        _inmodel_adapter = InModelAdapter(models_dir=str(MODELS_DIR))
    return _inmodel_adapter


def _load_user_input_adapter() -> UserInputAdapter:
    global _user_input_adapter
    if _user_input_adapter is None:
        models_dir = MODELS_DIR / "UserInput"
        _user_input_adapter = UserInputAdapter(models_dir=str(models_dir))
        if not getattr(_user_input_adapter, "available", False):
            print(
                f"[HydroScan] User input model unavailable: {_user_input_adapter.status}"
            )
    return _user_input_adapter


def _prepare_user_input_analysis(
    description: Optional[str],
    scene_majority: str,
    aggregation_result: Dict[str, Any],
    normalized_external: Dict[str, Any],
    metrics_per_frame: List[Dict[str, float]],
    media_info: Dict[str, int],
    classification_summary: Optional[Dict[str, Any]] = None,
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
        "classification_summary": classification_summary or {},
    }

    baseline_scores = compute_scores(score_context)

    assessment: Optional[UserInputAssessment] = None
    user_analysis: Optional[Dict[str, Any]] = None
    if description:
        adapter = _load_user_input_adapter()
        if getattr(adapter, "available", False) and getattr(adapter, "_llama", None):
            assessment = adapter.analyze(
                description,
                {
                    "detections": aggregation_result.get("detections", []),
                    "top_detection": aggregation_result.get("top_detection", {}),
                    "visual_metrics": aggregation_result.get("metrics_avg", {}),
                    "external": normalized_external,
                    "scene": scene_majority,
                    "base_scores": baseline_scores,
                    "classification_summary": classification_summary or {},
                },
            )
        else:
            reason = getattr(adapter, "status", "User input model unavailable")
            assessment = UserInputAssessment(
                available=False,
                conclusion="",
                score=45.0,
                confidence=35.0,
                rationale=None,
                model_name=getattr(adapter, "model_name", None),
                reason=reason or "User input model unavailable",
            )
        user_analysis = assessment.as_dict()
        score_context["user_text_assessment"] = user_analysis

    return score_context, baseline_scores, assessment, user_analysis


def _select_detector_adapter() -> RFDETRAdapter | InModelAdapter:
    backend = os.environ.get("MODEL_BACKEND", "rfdetr").lower()
    adapter = (
        _load_rfdetr_adapter() if backend != "inmodel" else _load_inmodel_adapter()
    )

    def _adapter_ready(candidate: RFDETRAdapter | InModelAdapter) -> bool:
        if hasattr(candidate, "model"):
            return getattr(candidate, "model") is not None
        models_loaded = getattr(candidate, "models_loaded", None)
        if isinstance(models_loaded, dict) and any(models_loaded.values()):
            return True
        if any(
            getattr(candidate, attr, None) is not None
            for attr in ("classifier", "obb_detector")
        ):
            return True
        return False

    if not _adapter_ready(adapter) and backend != "inmodel":
        fallback = _load_inmodel_adapter()
        if _adapter_ready(fallback):
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
    path: Path, max_frames: int = MAX_FRAMES_PER_VIDEO
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

    sampling_stride = max(1, total_frames // (max_frames * 2 or 1))
    keep_interval = max(1, total_frames // max_frames) if max_frames else 1

    frames_saved = 0
    prev_gray: Optional[np.ndarray] = None
    last_kept_idx = -keep_interval
    frame_index = 0

    while cap.isOpened() and frames_saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if sampling_stride > 1 and frame_index % sampling_stride != 0:
            frame_index += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_score = (
            _frame_difference_score(prev_gray, gray)
            if prev_gray is not None
            else FRAME_DIFF_THRESHOLD + 1.0
        )

        if (
            prev_gray is None
            or diff_score >= FRAME_DIFF_THRESHOLD
            or (frame_index - last_kept_idx) >= keep_interval
        ):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            frames_saved += 1
            prev_gray = gray
            last_kept_idx = frame_index

        frame_index += 1

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
    adapter_name = type(adapter).__name__
    print(f"[Analysis] Running {adapter_name} detection with threshold {threshold}")

    if isinstance(adapter, InModelAdapter):
        results = adapter.predict([image], conf=threshold)
        detections = results[0] if results else []
        print(f"[Analysis] {adapter_name} found {len(detections)} detections")
        return detections
    if hasattr(adapter, "predict"):
        detections = adapter.predict(image, threshold=threshold)  # type: ignore[arg-type]
        print(f"[Analysis] {adapter_name} found {len(detections or [])} detections")
        return detections or []
    raise RuntimeError("Adapter does not expose a compatible predict method")


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


def _frame_difference_score(prev_gray: np.ndarray, current_gray: np.ndarray) -> float:
    if prev_gray.shape != current_gray.shape:
        current_gray = cv2.resize(
            current_gray, (prev_gray.shape[1], prev_gray.shape[0])
        )
    diff = np.abs(prev_gray.astype(np.float32) - current_gray.astype(np.float32))
    return float(diff.mean())


def _create_debug_context(
    debug_enabled: bool, base_dir: Optional[Path]
) -> Dict[str, Any]:
    manifest = {category: [] for category in DEBUG_SNAPSHOT_CATEGORIES}
    dirs: Dict[str, Optional[Path]] = {}
    counts: defaultdict[str, int] = defaultdict(int)
    enabled = debug_enabled and base_dir is not None

    if enabled and base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)
        for category in DEBUG_SNAPSHOT_CATEGORIES:
            category_dir = base_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            dirs[category] = category_dir
    else:
        for category in DEBUG_SNAPSHOT_CATEGORIES:
            dirs[category] = None

    return {
        "enabled": enabled,
        "base_dir": base_dir,
        "dirs": dirs,
        "manifest": manifest,
        "counts": counts,
        "limit": MAX_DEBUG_IMAGES,
        "filter_saved_ops": set(),
        "scene_saved_indices": set(),
        "detector_failure_ops": set(),
        "errors": {"detector": []},
    }


def _save_debug_snapshot(
    context: Dict[str, Any],
    category: str,
    image: Image.Image,
    filename: str,
    metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not context.get("enabled"):
        return None

    if category not in DEBUG_SNAPSHOT_CATEGORIES:
        return None

    if context["counts"][category] >= context["limit"]:
        return None

    target_dir: Optional[Path] = context["dirs"].get(category)
    if target_dir is None:
        return None

    target_path = target_dir / filename
    try:
        image.save(target_path, format="JPEG", quality=90)
    except Exception:
        return None

    rel_path = f"{category}/{filename}"
    entry = {
        **metadata,
        "relative_path": rel_path,
        "filename": filename,
        "category": category,
    }
    context["manifest"][category].append(entry)
    context["counts"][category] += 1
    return entry


def _evaluate_detection_confidence(aggregation: Dict[str, Any]) -> Tuple[int, float]:
    detections = aggregation.get("detections", []) or []
    top = aggregation.get("top_detection") or {}
    top_score = float(top.get("score", 0.0) or 0.0)
    return len(detections), top_score


def _analyze_packaging(aggregation: Dict[str, Any]) -> Dict[str, Any]:
    detections: List[Dict[str, Any]] = aggregation.get("detections", []) or []
    packaging_hits: List[Dict[str, Any]] = []
    detected_brand: Optional[str] = None
    non_water_brand: Optional[str] = None

    for det in detections:
        name = (det.get("class_name") or "").lower()
        if not name:
            continue
        if any(keyword in name for keyword in PACKAGING_KEYWORDS):
            packaging_hits.append(det)
            for water_brand in KNOWN_WATER_BRANDS:
                if water_brand in name:
                    detected_brand = water_brand
                    break
            else:
                non_water_brand = name

    water_visible = any(
        "water" in (det.get("class_name") or "").lower() for det in detections
    )

    return {
        "packaging_present": bool(packaging_hits),
        "brand": detected_brand,
        "non_water_brand": non_water_brand,
        "water_visible": water_visible,
        "packaging_detections": packaging_hits,
    }


def _extract_packaging_crops(
    frames: Sequence[Image.Image],
    packaging_info: Dict[str, Any],
    limit: int = 4,
) -> List[Image.Image]:
    crops: List[Image.Image] = []
    if not packaging_info.get("packaging_present"):
        return crops

    for det in packaging_info.get("packaging_detections", []):
        occurrences = det.get("occurrences") or []
        if not occurrences:
            continue
        first = occurrences[0]
        frame_index = first.get("frame_index")
        bbox = det.get("bbox")
        if frame_index is None or bbox is None:
            continue
        if frame_index < 0 or frame_index >= len(frames):
            continue

        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        except Exception:
            continue

        frame = frames[frame_index]
        crop = frame.crop((max(0, x1), max(0, y1), max(x2, x1 + 1), max(y2, y1 + 1)))
        crops.append(crop)
        if len(crops) >= limit:
            break

    return crops


def _dynamic_detection_pipeline(
    frames: Sequence[Image.Image],
    detector: RFDETRAdapter | InModelAdapter,
    debug_ctx: Dict[str, Any],
    timeline: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, float]],
    Dict[str, Any],
    Dict[str, Any],
]:
    if not frames:
        empty_aggregation = aggregate([])
        return (
            [],
            [],
            empty_aggregation,
            {
                "applied": [],
                "target_score": TOP_DETECTION_TARGET,
                "min_detections": MIN_DETECTIONS_REQUIRED,
            },
        )

    metrics_per_frame = [compute_visual_metrics(frame) for frame in frames]
    frames_for_aggregation: List[Dict[str, Any]] = []

    operations: List[str] = ["original", "auto_white_balance", "contrast_stretch"]
    remaining_ops: List[str] = ["denoise", "sharpen", "gamma_1.2", "deglare"]
    processed_ops: set[str] = set()
    applied_ops: List[str] = []

    friendly_initial = ", ".join(get_filter_display_name(op) for op in operations)
    _append_timeline(
        timeline,
        "filter_strategy",
        "in-progress",
        f"Initial filters: {friendly_initial}",
    )
    _append_timeline(
        timeline,
        "detector_inference",
        "in-progress",
        "Running detection pipeline",
    )

    while True:
        progress_made = False
        for op in operations:
            if op in processed_ops:
                continue
            progress_made = True
            applied_ops.append(op)
            friendly_name = get_filter_display_name(op)
            if op != "original":
                _append_timeline(
                    timeline,
                    "filter_strategy",
                    "in-progress",
                    f"Applying {friendly_name}",
                )

            for idx, frame in enumerate(frames):
                if op == "original":
                    variant_img = frame.convert("RGB")
                    metrics = metrics_per_frame[idx]
                else:
                    variants = generate_variants(
                        frame,
                        max_variants=1,
                        operations=[op],
                        include_original=False,
                    )
                    if not variants:
                        continue
                    variant_img = variants[0][1]
                    metrics = compute_visual_metrics(variant_img)

                try:
                    print(
                        f"[Analysis] Processing frame {idx} with filter '{friendly_name}'"
                    )
                    detections = _run_detector(
                        detector, variant_img, DETECTOR_THRESHOLD
                    )
                    print(
                        f"[Analysis] Frame {idx} ({friendly_name}): {len(detections)} detections found"
                    )
                except Exception as exc:
                    detail = f"{friendly_name} failed for frame {idx}: {exc}"
                    if op not in debug_ctx["detector_failure_ops"]:
                        _append_timeline(
                            timeline,
                            "detector_inference",
                            "warning",
                            detail,
                        )
                        debug_ctx["detector_failure_ops"].add(op)
                    if debug_ctx.get("enabled"):
                        debug_ctx["errors"]["detector"].append(
                            {
                                "variant": op,
                                "frame_index": idx,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                    continue

                if (
                    debug_ctx.get("enabled")
                    and op != "original"
                    and op not in debug_ctx["filter_saved_ops"]
                ):
                    filename = f"filter_{_sanitize_slug(op)}.jpg"
                    saved = _save_debug_snapshot(
                        debug_ctx,
                        "filters",
                        variant_img,
                        filename,
                        {
                            "variant": op,
                            "label": friendly_name,
                        },
                    )
                    if saved:
                        debug_ctx["filter_saved_ops"].add(op)

                debug_image_rel_path: Optional[str] = None
                if debug_ctx.get("enabled") and detections:
                    filename = f"frame{idx:03d}_{_sanitize_slug(op)}.jpg"
                    annotated = _annotate_detections(variant_img, detections)
                    saved = _save_debug_snapshot(
                        debug_ctx,
                        "detector",
                        annotated,
                        filename,
                        {
                            "frame_index": idx,
                            "variant": op,
                            "detections": len(detections),
                        },
                    )
                    if saved:
                        debug_image_rel_path = saved["relative_path"]

                frames_for_aggregation.append(
                    {
                        "frame_index": idx,
                        "variant": op,
                        "metrics": metrics,
                        "detections": detections,
                        "debug_image": debug_image_rel_path,
                    }
                )

            processed_ops.add(op)

        print(
            f"[Analysis] Aggregating {len(frames_for_aggregation)} frames from {len(processed_ops)} processing operations"
        )
        aggregation_result = aggregate(frames_for_aggregation)
        detection_count, top_score = _evaluate_detection_confidence(aggregation_result)
        print(
            f"[Analysis] Aggregation complete: {detection_count} detections, top score: {top_score:.3f}"
        )

        if (
            detection_count >= MIN_DETECTIONS_REQUIRED
            and top_score >= TOP_DETECTION_TARGET
        ):
            _append_timeline(
                timeline,
                "detector_inference",
                "done",
                f"Detections={detection_count}, top={top_score:.2f}",
            )
            break

        if not remaining_ops:
            status = "warning" if detection_count == 0 else "in-progress"
            _append_timeline(
                timeline,
                "detector_inference",
                status,
                f"Detections={detection_count}, top={top_score:.2f}",
            )
            break

        next_op = remaining_ops.pop(0)
        operations.append(next_op)
        _append_timeline(
            timeline,
            "filter_strategy",
            "in-progress",
            f"Boosting confidence with {get_filter_display_name(next_op)}",
        )

        if not progress_made:
            _append_timeline(
                timeline,
                "detector_inference",
                "warning",
                "Unable to apply additional filters",
            )
            break

    filter_summary = {
        "applied": applied_ops,
        "target_score": TOP_DETECTION_TARGET,
        "min_detections": MIN_DETECTIONS_REQUIRED,
    }

    return frames_for_aggregation, metrics_per_frame, aggregation_result, filter_summary


@router.post("/analyze")
async def analyze_endpoint(
    files: List[UploadFile] = File(..., description="Images and/or videos"),
    description: Optional[str] = Form(
        None, description="User description: smell/color/feel/temperature"
    ),
    lat: Optional[float] = Form(None, description="Latitude for external data"),
    lon: Optional[float] = Form(None, description="Longitude for external data"),
    debug: bool = Form(False),
    dont_save: bool = Form(
        False, description="Skip persisting this analysis to history"
    ),
):
    if not files:
        raise HTTPException(status_code=400, detail="No media uploaded")

    if len(files) > MAX_IMAGES + MAX_VIDEOS:
        raise HTTPException(status_code=400, detail="Too many files uploaded")

    image_candidates: List[UploadFile] = []
    video_candidates: List[UploadFile] = []
    for upload in files:
        if _is_video(upload):
            video_candidates.append(upload)
        else:
            image_candidates.append(upload)

    if len(image_candidates) > MAX_IMAGES:
        raise HTTPException(
            status_code=400, detail=f"Maximum {MAX_IMAGES} images allowed"
        )
    if len(video_candidates) > MAX_VIDEOS:
        raise HTTPException(
            status_code=400, detail=f"Maximum {MAX_VIDEOS} videos allowed"
        )

    history_enabled = not dont_save
    analysis_id = str(uuid.uuid4())
    storage_dir = (
        HISTORY_DIR / analysis_id
        if history_enabled
        else Path(tempfile.mkdtemp(prefix="hydroscan-"))
    )
    _ensure_history_dir(storage_dir)

    debug_base_dir = storage_dir / "debug" if (debug and history_enabled) else None
    debug_ctx = _create_debug_context(debug, debug_base_dir)

    saved_files: List[Dict[str, Any]] = []
    actual_image_paths: List[Path] = []
    actual_video_paths: List[Path] = []

    try:
        for upload in files:
            dest = await _save_upload(upload, storage_dir)
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

        timeline: List[Dict[str, Any]] = []
        _append_timeline(
            timeline,
            "preparing_media",
            "in-progress",
            f"{len(actual_image_paths)} images, {len(actual_video_paths)} videos",
        )

        frames: List[Image.Image] = []
        for img_path in actual_image_paths:
            frames.append(_load_image(img_path))

        for video_path in actual_video_paths:
            try:
                video_frames = _extract_frames_from_video(video_path)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            frames.extend(video_frames)

        if not frames:
            raise HTTPException(
                status_code=400, detail="Unable to decode any frames from uploads"
            )

        _append_timeline(
            timeline,
            "preparing_media",
            "done",
            f"Frames retained: {len(frames)}",
            progress=1,
            total_steps=8,
        )

        # Enhanced Scene Detection with Filter Stack
        _append_timeline(
            timeline,
            "scene_detection",
            "in-progress",
            "🔍 Applying filter stack for better scene analysis...",
            progress=2,
            total_steps=8,
        )

        place365 = _load_place_adapter()
        sampled_scene_frames = frames[:SCENE_SAMPLE_FRAMES] or frames[:1]

        # Apply filter variants to improve scene detection accuracy
        scene_results_all = []
        best_scene_results = []

        for i, frame in enumerate(sampled_scene_frames):
            _append_timeline(
                timeline,
                "scene_detection",
                "in-progress",
                f"🎛️ Generating filter variants for frame {i+1}/{len(sampled_scene_frames)}...",
                progress=2,
                total_steps=8,
            )

            # Generate filter variants for better scene analysis
            filter_variants = generate_variants(
                frame,
                max_variants=4,  # Keep reasonable for scene detection
                operations=[
                    "original",
                    "auto_white_balance",
                    "contrast_stretch",
                    "gamma_adaptive",
                ],
                include_original=True,
            )

            frame_results = []
            for filter_name, variant_frame in filter_variants:
                _append_timeline(
                    timeline,
                    "scene_detection",
                    "in-progress",
                    f"📊 Analyzing with {get_filter_display_name(filter_name)}...",
                    progress=2,
                    total_steps=8,
                )
                result = place365.classify(variant_frame)
                result["filter_applied"] = filter_name
                result["filter_display"] = get_filter_display_name(filter_name)
                frame_results.append(result)
                scene_results_all.append(result)

            # Select best result for this frame (highest confidence)
            best_result = max(frame_results, key=lambda x: x.get("confidence", 0.0))
            best_scene_results.append(best_result)

            _append_timeline(
                timeline,
                "scene_detection",
                "in-progress",
                f"✅ Best for frame {i+1}: {best_result.get('scene', 'unknown')} "
                f"({best_result.get('confidence', 0.0):.1%}) via {best_result.get('filter_display', 'Original')}",
                progress=2,
                total_steps=8,
            )

        # Determine scene with 50% confidence threshold
        scene_labels = [res.get("scene", "unknown") for res in best_scene_results]
        scene_confidences = [res.get("confidence", 0.0) for res in best_scene_results]
        scene_majority = _majority_vote(scene_labels)
        mean_scene_conf = (
            sum(scene_confidences) / len(scene_confidences)
            if scene_confidences
            else 0.0
        )

        # Apply 50% confidence threshold
        confident_outdoor = mean_scene_conf >= 0.5 and scene_majority == "outdoor"
        confident_indoor = mean_scene_conf >= 0.5 and scene_majority == "indoor"

        if confident_outdoor:
            final_scene = "outdoor"
            confidence_status = "high"
        elif confident_indoor:
            final_scene = "indoor"
            confidence_status = "high"
        else:
            # Below 50% confidence - default to indoor workflow for safety
            final_scene = "indoor"
            confidence_status = "low"
            _append_timeline(
                timeline,
                "scene_detection",
                "warning",
                f"⚠️ Low confidence ({mean_scene_conf:.1%}) - defaulting to indoor workflow",
                progress=2,
                total_steps=8,
            )

        # Log the best filter variant used
        best_overall = max(best_scene_results, key=lambda x: x.get("confidence", 0.0))
        best_filter_used = best_overall.get("filter_display", "Original")

        _append_timeline(
            timeline,
            "scene_detection",
            "done",
            f"🎯 Scene: {final_scene.title()} (confidence: {mean_scene_conf:.1%}, "
            f"best filter: {best_filter_used})",
            progress=2,
            total_steps=8,
        )

        # Scene Routing with Enhanced Workflow Logic
        scene_majority = final_scene  # Use the confidence-adjusted scene
        if scene_majority == "outdoor":
            _append_timeline(
                timeline,
                "scene_routing",
                "done",
                "🌳 Outdoor detected → Location-based data will be fetched",
                progress=3,
                total_steps=8,
            )
        elif scene_majority == "indoor":
            _append_timeline(
                timeline,
                "scene_routing",
                "done",
                "🏠 Indoor detected → Object detection for packaging analysis",
                progress=3,
                total_steps=8,
            )
        else:
            _append_timeline(
                timeline,
                "scene_routing",
                "done",
                "❓ Unknown scene → Using indoor workflow with water detection",
                progress=3,
                total_steps=8,
            )

        if debug_ctx.get("enabled"):
            for idx, (frame, result) in enumerate(
                zip(sampled_scene_frames, best_scene_results)
            ):
                label = result.get("label") or result.get("scene") or "unknown"
                confidence = float(result.get("confidence", 0.0) or 0.0)
                annotated = frame.convert("RGB")
                draw = ImageDraw.Draw(annotated)
                text = f"{label} ({confidence:.2f})"
                try:
                    bbox = draw.textbbox((16, 16), text)
                except Exception:
                    bbox = (16, 16, 16 + len(text) * 7, 36)
                draw.rectangle(
                    (bbox[0] - 8, bbox[1] - 8, bbox[2] + 8, bbox[3] + 8),
                    fill=(15, 23, 42, 200),
                )
                draw.text((bbox[0], bbox[1]), text, fill=(240, 249, 255))
                _save_debug_snapshot(
                    debug_ctx,
                    "scene",
                    annotated,
                    f"scene_{idx:02d}.jpg",
                    {"frame_index": idx, "label": label, "confidence": confidence},
                )

        detector = _select_detector_adapter()
        normalized_external: Dict[str, Any] = {}
        external_raw: Optional[Dict[str, Any]] = None
        packaging_info: Dict[str, Any] = {}
        packaging_crops: List[Image.Image] = []
        packaging_water_hits = 0
        non_water_brand_detected = False

        if scene_majority == "outdoor":
            if lat is not None and lon is not None:
                _append_timeline(
                    timeline,
                    "location_data",
                    "in-progress",
                    "📍 Fetching location-based water quality data...",
                    progress=4,
                    total_steps=8,
                )
                try:
                    external_raw = fetch_nearest_station(lat=lat, lon=lon)
                    normalized_external = normalize_external(external_raw)
                    if normalized_external:
                        distance = normalized_external.get("distance_km")
                        if isinstance(distance, (int, float)):
                            distance_detail = f"{float(distance):.2f} km"
                        else:
                            distance_detail = "unknown distance"
                        station_id = normalized_external.get("station_id") or "unknown"
                        _append_timeline(
                            timeline,
                            "location_data",
                            "done",
                            f"✅ Station found: {station_id} ({distance_detail} away)",
                            progress=4,
                            total_steps=8,
                        )
                        _append_timeline(
                            timeline,
                            "detector_selection",
                            "done",
                            "🔍 Using RF-DETR detector with location data support",
                            progress=5,
                            total_steps=8,
                        )
                    else:
                        _append_timeline(
                            timeline,
                            "location_data",
                            "warning",
                            "⚠️ No station found within range",
                            progress=4,
                            total_steps=8,
                        )
                        fallback = _load_inmodel_adapter()
                        if (
                            fallback.models_loaded["classification"]
                            or fallback.models_loaded["obb"]
                        ):
                            detector = fallback
                            _append_timeline(
                                timeline,
                                "detector_selection",
                                "warning",
                                "🏠 Falling back to in-house model (no location data)",
                                progress=5,
                                total_steps=8,
                            )
                        else:
                            _append_timeline(
                                timeline,
                                "outdoor_detector",
                                "error",
                                "In-house detector unavailable",
                            )
                except Exception as exc:
                    _append_timeline(timeline, "outdoor_external", "error", str(exc))
                    fallback = _load_inmodel_adapter()
                    if (
                        fallback.models_loaded["classification"]
                        or fallback.models_loaded["obb"]
                    ):
                        detector = fallback
                        _append_timeline(
                            timeline,
                            "outdoor_detector",
                            "warning",
                            "In-house detector engaged after external data failure",
                        )
                    else:
                        _append_timeline(
                            timeline,
                            "outdoor_detector",
                            "error",
                            "In-house detector unavailable",
                        )
            else:
                # No location provided for outdoor scene
                _append_timeline(
                    timeline,
                    "location_data",
                    "warning",
                    "📍 Location not provided → Cannot fetch location-based data",
                    progress=4,
                    total_steps=8,
                )
                fallback = _load_inmodel_adapter()
                if (
                    fallback.models_loaded["classification"]
                    or fallback.models_loaded["obb"]
                ):
                    detector = fallback
                    _append_timeline(
                        timeline,
                        "detector_selection",
                        "warning",
                        "🏠 Using in-house detector (no location data available)",
                        progress=5,
                        total_steps=8,
                    )
                else:
                    _append_timeline(
                        timeline,
                        "detector_selection",
                        "error",
                        "❌ In-house detector unavailable",
                        progress=5,
                        total_steps=8,
                    )
        else:
            # Enhanced Indoor/Unknown Workflow
            _append_timeline(
                timeline,
                "object_detection",
                "in-progress",
                "🔍 Running object detection model...",
                progress=4,
                total_steps=8,
            )

        # Run detection pipeline
        (
            frames_for_aggregation,
            metrics_per_frame,
            aggregation_result,
            filter_summary,
        ) = _dynamic_detection_pipeline(frames, detector, debug_ctx, timeline)

        friendly_filters = ", ".join(
            get_filter_display_name(op) for op in filter_summary.get("applied", [])
        )
        if friendly_filters:
            _append_timeline(
                timeline,
                "adaptive_filters",
                "done",
                f"🎛️ Applied filters: {friendly_filters}",
                progress=5,
                total_steps=8,
            )

        aggregated_detections = aggregation_result.get("detections", []) or []
        top_detection = aggregation_result.get("top_detection")
        detection_detail = f"Found {len(aggregated_detections)} object(s)"
        if top_detection:
            detection_detail += f" → Top: {top_detection.get('class_name')} ({float(top_detection.get('score', 0.0)):.1%})"

        _append_timeline(
            timeline,
            "object_detection",
            "done",
            detection_detail,
            progress=6,
            total_steps=8,
        )

        # Enhanced Indoor Workflow Logic
        if scene_majority != "outdoor":
            packaging_info = _analyze_packaging(aggregation_result)

            # Check for recognizable branded packaging
            if packaging_info.get("packaging_present"):
                _append_timeline(
                    timeline,
                    "packaging_analysis",
                    "done",
                    f"📦 Packaging detected: {len(packaging_info['packaging_detections'])} items",
                    progress=6,
                    total_steps=8,
                )

                detected_brand = packaging_info.get("brand")
                non_water_brand = packaging_info.get("non_water_brand")

                if detected_brand:
                    # Recognizable water brand found
                    _append_timeline(
                        timeline,
                        "brand_ocr",
                        "done",
                        f"💧 Water brand detected: {detected_brand.title()}",
                        progress=7,
                        total_steps=8,
                    )
                    # TODO: Fetch brand quality data if available

                elif non_water_brand and not _contains_water_keyword(non_water_brand):
                    # Non-water brand detected - error case
                    non_water_brand_detected = True
                    _append_timeline(
                        timeline,
                        "brand_ocr",
                        "error",
                        f"❌ Non-water brand detected: {non_water_brand}",
                        progress=7,
                        total_steps=8,
                    )

                else:
                    # Brand unknown - crop to water area if transparent
                    _append_timeline(
                        timeline,
                        "brand_ocr",
                        "warning",
                        "❓ Brand unknown → Isolating liquid area in packaging",
                        progress=7,
                        total_steps=8,
                    )

                    packaging_crops = _extract_packaging_crops(frames, packaging_info)
                    if packaging_crops and debug_ctx.get("enabled"):
                        for idx, crop in enumerate(packaging_crops):
                            _save_debug_snapshot(
                                debug_ctx,
                                "ocr",
                                crop,
                                f"packaging_crop_{idx:02d}.jpg",
                                {"crop_index": idx, "source": "packaging_isolation"},
                            )

                    if packaging_crops:
                        _append_timeline(
                            timeline,
                            "liquid_isolation",
                            "in-progress",
                            "🔬 Analyzing isolated liquid area with in-house model...",
                            progress=7,
                            total_steps=8,
                        )

                        fallback_detector = _load_inmodel_adapter()
                        if (
                            fallback_detector.models_loaded["classification"]
                            or fallback_detector.models_loaded["obb"]
                        ):
                            crop_results = fallback_detector.predict(
                                list(packaging_crops), conf=DETECTOR_THRESHOLD
                            )
                            for crop_det in crop_results:
                                if any(
                                    _contains_water_keyword(det.get("class_name"))
                                    for det in crop_det
                                ):
                                    packaging_water_hits += 1

                            if packaging_water_hits:
                                _append_timeline(
                                    timeline,
                                    "liquid_isolation",
                                    "done",
                                    f"✅ Water detected in {packaging_water_hits} packaging crop(s)",
                                    progress=8,
                                    total_steps=8,
                                )
                            else:
                                _append_timeline(
                                    timeline,
                                    "liquid_isolation",
                                    "warning",
                                    "⚠️ Liquid not clearly visible in packaging",
                                    progress=8,
                                    total_steps=8,
                                )
                    else:
                        _append_timeline(
                            timeline,
                            "liquid_isolation",
                            "error",
                            "❌ Cannot isolate liquid area from packaging",
                            progress=7,
                            total_steps=8,
                        )

            else:
                # No packaging detected - try direct water detection
                _append_timeline(
                    timeline,
                    "packaging_analysis",
                    "warning",
                    "📦 No recognizable packaging found",
                    progress=6,
                    total_steps=8,
                )

                # Try to detect water directly in the images/video frames
                water_detections = [
                    det
                    for det in aggregated_detections
                    if _contains_water_keyword(det.get("class_name", ""))
                ]

                if water_detections:
                    _append_timeline(
                        timeline,
                        "direct_water_detection",
                        "done",
                        f"💧 Water detected directly in {len(water_detections)} detection(s)",
                        progress=8,
                        total_steps=8,
                    )
                else:
                    _append_timeline(
                        timeline,
                        "direct_water_detection",
                        "error",
                        "❌ No water detected in provided images/video",
                        progress=8,
                        total_steps=8,
                    )

        # Get detailed In House Model predictions with bounding boxes
        custom_model_predictions = []
        classification_summary = {"available": False, "samples": []}
        custom_model_adapter = _load_inmodel_adapter()
        if (
            custom_model_adapter.models_loaded["classification"]
            or custom_model_adapter.models_loaded["obb"]
        ):
            _append_timeline(
                timeline,
                "custom_model_analysis",
                "in-progress",
                "🤖 Running In House Model for detailed predictions...",
                progress=7,
                total_steps=8,
            )

            try:
                # Run comprehensive predictions on representative frames
                prediction_frames = frames[:3] if len(frames) > 3 else frames

                # Run traditional prediction for compatibility
                frame_predictions = custom_model_adapter.predict(
                    prediction_frames, conf=DETECTOR_THRESHOLD
                )

                for frame_idx, predictions in enumerate(frame_predictions):
                    for pred in predictions:
                        pred_with_frame = pred.copy()
                        pred_with_frame["frame_index"] = frame_idx
                        custom_model_predictions.append(pred_with_frame)

                # Run comprehensive YOLOv11 predictions with OBB and classification
                for frame_idx, frame in enumerate(prediction_frames):
                    try:
                        comprehensive_results = (
                            custom_model_adapter.predict_comprehensive(
                                frame, conf=DETECTOR_THRESHOLD
                            )
                        )

                        # Add OBB detections
                        for obb_detection in comprehensive_results.get(
                            "obb_detections", []
                        ):
                            obb_detection["frame_index"] = frame_idx
                            obb_detection["detection_type"] = "obb"
                            custom_model_predictions.append(obb_detection)

                        # Add classification results
                        classifications_payload = comprehensive_results.get(
                            "classifications", []
                        )
                        if not classifications_payload and comprehensive_results.get(
                            "classification"
                        ):
                            classifications_payload = [
                                comprehensive_results["classification"]
                            ]

                        for classification in classifications_payload:
                            classification_entry = classification.copy()
                            classification_entry["frame_index"] = frame_idx
                            classification_entry["detection_type"] = "classification"
                            custom_model_predictions.append(classification_entry)

                    except Exception as comp_e:
                        print(
                            f"[HydroScan] Comprehensive prediction error for frame {frame_idx}: {comp_e}"
                        )

                _append_timeline(
                    timeline,
                    "custom_model_analysis",
                    "done",
                    f"🤖 Custom Model: {len(custom_model_predictions)} predictions across {len(prediction_frames)} frames",
                    progress=7,
                    total_steps=8,
                )

            except Exception as e:
                _append_timeline(
                    timeline,
                    "custom_model_analysis",
                    "warning",
                    f"⚠️ In House Model error: {str(e)}",
                    progress=7,
                    total_steps=8,
                )

        classification_summary = _summarize_classifications(custom_model_predictions)

        # Water confirmation logic (applies to all scenarios)
        if scene_majority == "outdoor":
            total_detections = len(aggregated_detections) + len(
                custom_model_predictions
            )
            if aggregated_detections or custom_model_predictions:
                _append_timeline(
                    timeline,
                    "water_confirmation",
                    "done",
                    f"Detector observed {total_detections} detection(s) outdoors",
                )
            else:
                _append_timeline(
                    timeline,
                    "water_confirmation",
                    "warning",
                    "Detector did not observe water cues outdoors",
                )

        water_detected = any(
            _contains_water_keyword(det.get("class_name"))
            for det in aggregated_detections
        )

        # Also check custom model predictions for water-related detections
        if custom_model_predictions:
            water_detected = water_detected or any(
                _contains_water_keyword(pred.get("class_name") or pred.get("class"))
                or "water"
                in (pred.get("class_name") or pred.get("class") or "").lower()
                or pred.get("detection_type")
                == "classification"  # YOLOv11 water quality classification counts as water detection
                for pred in custom_model_predictions
            )

        if packaging_info.get("water_visible"):
            water_detected = True
        if packaging_water_hits > 0:
            water_detected = True

        media_info = {
            "media_count": len(actual_image_paths) + len(actual_video_paths),
            "variant_count": len(frames_for_aggregation),
            "frame_count": len(frames),
        }

        if non_water_brand_detected:
            message = (
                "Detected branded packaging that is not water-related. "
                "Please provide water-specific samples."
            )
            _append_timeline(
                timeline,
                "finalizing",
                "done",
                "Detected non-water brand; analysis halted",
            )
            result_payload = {
                "analysis_id": analysis_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "description": description,
                "scene": {
                    "majority": scene_majority,
                    "samples": best_scene_results,
                    "confidence": mean_scene_conf,
                    "filter_analysis": scene_results_all,
                },
                "external_data": normalized_external,
                "media": {
                    "saved_files": saved_files,
                    "frame_count": len(frames),
                    "variant_count": len(frames_for_aggregation),
                },
                "aggregation": aggregation_result,
                "custom_model_predictions": custom_model_predictions,
                "classification_summary": classification_summary,
                "timeline": timeline,
                "status": "non_water_brand",
                "message": message,
                "packaging": packaging_info,
                "history_saved": history_enabled,
                "debug": {
                    "enabled": debug_ctx.get("enabled"),
                    "snapshots": (
                        debug_ctx["manifest"] if debug_ctx.get("enabled") else {}
                    ),
                    "errors": debug_ctx["errors"],
                },
            }

            if history_enabled:
                result_path = storage_dir / "result.json"
                with result_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        _serialize_for_json(result_payload),
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                if debug_ctx.get("enabled"):
                    debug_payload = {
                        "frames": frames_for_aggregation,
                        "metrics": metrics_per_frame,
                        "filter_summary": filter_summary,
                        "packaging": packaging_info,
                        "snapshots": debug_ctx["manifest"],
                        "errors": debug_ctx["errors"],
                    }
                    debug_path = storage_dir / "debug.json"
                    with debug_path.open("w", encoding="utf-8") as f:
                        json.dump(
                            _serialize_for_json(debug_payload),
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )

            response: Dict[str, Any] = {
                "analysis_id": analysis_id,
                "status": "non_water_brand",
                "message": message,
                "timeline": timeline,
                "packaging": packaging_info,
                "classification_summary": classification_summary,
                "history_saved": history_enabled,
            }
            if not history_enabled:
                response["result"] = result_payload
            return response

        if not water_detected:
            # Provide more specific error message based on scene type with debug info
            debug_info = (
                f" [Debug: {len(aggregated_detections)} aggregated detections, {len(custom_model_predictions)} custom model predictions]"
                if debug
                else ""
            )

            if scene_majority == "outdoor":
                message = (
                    "Water wasn't detected in the outdoor scene. "
                    "Please ensure water bodies (lakes, rivers, streams) are clearly visible, "
                    "or try different camera angles and lighting conditions."
                    + debug_info
                )
            else:
                if packaging_info.get("packaging_present"):
                    if packaging_info.get("brand"):
                        message = (
                            f"Detected branded packaging ({packaging_info['brand']}) but couldn't "
                            "confirm water content. Please ensure the liquid is clearly visible through "
                            "transparent packaging."
                        )
                    else:
                        message = (
                            "Detected packaging but couldn't identify the brand or confirm water content. "
                            "Please provide clearer images of the label or ensure the liquid is visible."
                        )
                else:
                    message = (
                        "No water or water-related packaging detected in indoor/unknown scene. "
                        "Please provide images showing water directly, in transparent containers, "
                        "or with visible packaging labels."
                    )

            _append_timeline(
                timeline,
                "water_confirmation",
                "error",
                message,
            )
            user_analysis: Optional[Dict[str, Any]] = None
            assessment: Optional[UserInputAssessment] = None
            if description:
                _append_timeline(
                    timeline,
                    "user_input_analysis",
                    "in-progress",
                    "Assessing user notes with LLaMA 3",
                )
                _, _, assessment, user_analysis = _prepare_user_input_analysis(
                    description,
                    scene_majority,
                    aggregation_result,
                    normalized_external,
                    metrics_per_frame,
                    media_info,
                    classification_summary,
                )
                if assessment and assessment.available:
                    _append_timeline(
                        timeline,
                        "user_input_analysis",
                        "done",
                        f"LLM confidence {assessment.confidence:.0f}%",
                    )
                else:
                    _append_timeline(
                        timeline,
                        "user_input_analysis",
                        "warning",
                        (
                            assessment.reason
                            if assessment and assessment.reason
                            else "User input model unavailable"
                        ),
                    )

            result_payload = {
                "analysis_id": analysis_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "description": description,
                "scene": {
                    "majority": scene_majority,
                    "samples": best_scene_results,
                    "confidence": mean_scene_conf,
                    "filter_analysis": scene_results_all,
                },
                "media": {
                    "saved_files": saved_files,
                    "frame_count": len(frames),
                    "variant_count": len(frames_for_aggregation),
                },
                "aggregation": aggregation_result,
                "custom_model_predictions": custom_model_predictions,
                "classification_summary": classification_summary,
                "timeline": timeline,
                "status": "no_water_detected",
                "message": message,
                "user_analysis": user_analysis,
                "external_data": normalized_external,
                "packaging": packaging_info,
                "filter_summary": filter_summary,
                "history_saved": history_enabled,
                "debug": {
                    "enabled": debug_ctx.get("enabled"),
                    "snapshots": (
                        debug_ctx["manifest"] if debug_ctx.get("enabled") else {}
                    ),
                    "errors": debug_ctx["errors"],
                },
            }

            if history_enabled:
                result_path = storage_dir / "result.json"
                with result_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        _serialize_for_json(result_payload),
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                if debug_ctx.get("enabled"):
                    debug_payload = {
                        "frames": frames_for_aggregation,
                        "metrics": metrics_per_frame,
                        "filter_summary": filter_summary,
                        "packaging": packaging_info,
                        "snapshots": debug_ctx["manifest"],
                        "errors": debug_ctx["errors"],
                    }
                    debug_path = storage_dir / "debug.json"
                    with debug_path.open("w", encoding="utf-8") as f:
                        json.dump(
                            _serialize_for_json(debug_payload),
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )

            response = {
                "analysis_id": analysis_id,
                "status": "no_water_detected",
                "message": message,
                "timeline": timeline,
                "aggregation": aggregation_result,
                "custom_model_predictions": custom_model_predictions,
                "classification_summary": classification_summary,
                "history_saved": history_enabled,
            }
            if user_analysis is not None:
                response["user_analysis"] = user_analysis
            if not history_enabled:
                response["result"] = result_payload
            return response

        if description:
            _append_timeline(
                timeline,
                "user_input_analysis",
                "in-progress",
                "Assessing user notes with LLaMA 3",
            )

        score_context, _baseline_scores, assessment, user_analysis = (
            _prepare_user_input_analysis(
                description,
                scene_majority,
                aggregation_result,
                normalized_external,
                metrics_per_frame,
                media_info,
                classification_summary,
            )
        )

        if description:
            if assessment and assessment.available:
                _append_timeline(
                    timeline,
                    "user_input_analysis",
                    "done",
                    f"LLM confidence {assessment.confidence:.0f}%",
                )
            else:
                _append_timeline(
                    timeline,
                    "user_input_analysis",
                    "warning",
                    (
                        assessment.reason
                        if assessment and assessment.reason
                        else "User input model unavailable"
                    ),
                )

        scores = compute_scores(score_context)
        _append_timeline(timeline, "scoring", "done", scores["band_label"])

        result_payload = {
            "analysis_id": analysis_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "description": description,
            "scene": {
                "majority": scene_majority,
                "samples": best_scene_results,
                "confidence": mean_scene_conf,
                "filter_analysis": scene_results_all,
            },
            "external_data": normalized_external,
            "media": {
                "saved_files": saved_files,
                "frame_count": len(frames),
                "variant_count": len(frames_for_aggregation),
            },
            "aggregation": aggregation_result,
            "custom_model_predictions": custom_model_predictions,
            "scores": scores,
            "timeline": timeline,
            "user_analysis": user_analysis,
            "packaging": packaging_info,
            "packaging_water_hits": packaging_water_hits,
            "filter_summary": filter_summary,
            "classification_summary": classification_summary,
            "history_saved": history_enabled,
            "debug": {
                "enabled": debug_ctx.get("enabled"),
                "snapshots": debug_ctx["manifest"] if debug_ctx.get("enabled") else {},
                "errors": debug_ctx["errors"],
            },
        }

        _append_timeline(
            timeline,
            "finalizing",
            "done",
            "Results saved" if history_enabled else "Results prepared",
        )

        if history_enabled:
            result_path = storage_dir / "result.json"
            with result_path.open("w", encoding="utf-8") as f:
                json.dump(
                    _serialize_for_json(result_payload),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            if debug_ctx.get("enabled"):
                debug_payload = {
                    "frames": frames_for_aggregation,
                    "metrics": metrics_per_frame,
                    "filter_summary": filter_summary,
                    "packaging": packaging_info,
                    "snapshots": debug_ctx["manifest"],
                    "errors": debug_ctx["errors"],
                }
                debug_path = storage_dir / "debug.json"
                with debug_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        _serialize_for_json(debug_payload),
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

        response: Dict[str, Any] = {
            "analysis_id": analysis_id,
            "status": "completed",
            "scores": scores,
            "scene": result_payload["scene"],
            "external_data": normalized_external,
            "timeline": timeline,
            "media": result_payload["media"],
            "aggregation": aggregation_result,
            "custom_model_predictions": custom_model_predictions,
            "user_analysis": user_analysis,
            "filter_summary": filter_summary,
            "packaging": packaging_info,
            "packaging_water_hits": packaging_water_hits,
            "classification_summary": classification_summary,
            "history_saved": history_enabled,
        }

        if debug_ctx.get("enabled"):
            base_url = f"/api/results/{analysis_id}/artifacts"
            detector_snapshots = [
                {
                    **entry,
                    "url": "{}/{}".format(
                        base_url,
                        str(entry.get("relative_path", "")).replace("\\", "/"),
                    ),
                }
                for entry in debug_ctx["manifest"].get("detector", [])
            ]
            response["debug_images"] = detector_snapshots
            response["detector_errors"] = debug_ctx["errors"]["detector"]
            if history_enabled:
                response["debug_path"] = str(storage_dir / "debug.json")

        if history_enabled:
            response["result_path"] = str(storage_dir / "result.json")
        else:
            response["result"] = result_payload

        return response
    finally:
        if not history_enabled:
            shutil.rmtree(storage_dir, ignore_errors=True)
