from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_POTABILITY_WEIGHTS: Dict[str, float] = {
    "external": 30.0,
    "visual": 20.0,
    "model_confidence": 15.0,
    "color": 10.0,
    "user_text": 10.0,
    "temporal": 10.0,
    "corroboration": 5.0,
    "custom_classification": 15.0,
}

DEFAULT_CONFIDENCE_WEIGHTS: Dict[str, float] = {
    "detector_confidence": 40.0,
    "image_quality": 20.0,
    "media_corroboration": 15.0,
    "external": 15.0,
    "user_text": 10.0,
    "custom_classification": 10.0,
}

DIRTY_KEYWORDS = {
    "dirty",
    "murky",
    "pollut",
    "foam",
    "oil",
    "sheen",
    "algae",
    "scum",
    "trash",
    "waste",
    "sewage",
    "contamin",
}


@dataclass(slots=True)
class ComponentScore:
    value: float
    weight: float

    def weighted(self) -> float:
        return self.value * (self.weight / 100.0)


def _band_from_score(score: float) -> str:
    if score >= 100.0:
        return "Drinkable"
    if 51.0 <= score < 100.0:
        return "Very clean, not drinkable"
    if score >= 50.0:
        return "Clean for daily use, not drinkable"
    if score >= 26.0:
        return "Less clean"
    return "Unclean"


def _normalize_weights(
    weights: Dict[str, float], available: Iterable[str]
) -> Dict[str, float]:
    present_keys: List[str] = [k for k in available if weights.get(k, 0.0) > 0.0]
    if not present_keys:
        return {key: 0.0 for key in weights}
    total = sum(weights[key] for key in present_keys)
    if total <= 0.0:
        return {key: 0.0 for key in weights}
    return {
        key: (weights[key] / total) * 100.0 if key in present_keys else 0.0
        for key in weights
    }


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return float(max(lower, min(upper, value)))


def _normalize_probability(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0.0:
        numeric = 0.0
    if numeric > 1.5:
        numeric = numeric / 100.0
    if numeric > 1.0:
        numeric = 1.0
    return numeric


def _score_user_text(text: str | None) -> Tuple[float, float]:
    if not text:
        return 50.0, 60.0  # neutral impact, moderate confidence contribution
    text_lower = text.lower()
    negatives = [
        "smell",
        "odor",
        "brown",
        "dirty",
        "muddy",
        "cloudy",
        "polluted",
        "foam",
    ]
    positives = ["clear", "no smell", "fresh", "clean"]
    score = 50.0
    for token in positives:
        if token in text_lower:
            score += 10
    for token in negatives:
        if token in text_lower:
            score -= 15
    return _clamp(score), _clamp(60.0 + (score - 50.0))


def _score_external(external: Dict[str, Any] | None) -> Tuple[float, float]:
    if not external:
        return 0.0, 0.0

    quality = external.get("overall_quality", 0.0)
    recency_days = external.get("recency_days")
    if recency_days is not None:
        recency_penalty = min(max(recency_days, 0.0) / 365.0, 1.0) * 30.0
        quality = max(0.0, quality - recency_penalty)
    return _clamp(quality), _clamp(100.0 - (recency_days or 0.0))


def _score_visual_metrics(visual_avg: Dict[str, float]) -> Tuple[float, float]:
    turbidity = visual_avg.get("turbidity_proxy")
    foam = visual_avg.get("foam_proxy")
    edge_density = visual_avg.get("edge_density")

    if turbidity is None:
        return 50.0, 50.0

    clarity = 100.0 - turbidity
    foam_penalty = (foam or 0.0) * 0.3
    clutter_penalty = max(0.0, (edge_density or 0.0) - 40.0) * 0.25
    visual_score = clarity - foam_penalty - clutter_penalty
    quality_score = _clamp(clarity)
    return _clamp(visual_score), _clamp(quality_score)


def _score_color(visual_avg: Dict[str, float]) -> float:
    deviation = visual_avg.get("color_deviation")
    if deviation is None:
        return 50.0
    return _clamp(100.0 - deviation)


def _score_model_confidence(detections: List[Dict[str, Any]]) -> float:
    if not detections:
        return 40.0

    scores: List[float] = []
    for det in detections:
        value = det.get("score")
        if value is None:
            value = det.get("confidence")
        prob = _normalize_probability(value)
        if prob is None:
            continue
        scores.append(_clamp(prob * 100.0))

    if not scores:
        return 40.0

    top_score = max(scores)
    median_score = median(scores)
    blended = 0.7 * top_score + 0.3 * median_score
    return _clamp(blended)


def _dirty_signal_from_detections(detections: List[Dict[str, Any]]) -> float:
    best = 0.0
    for det in detections:
        name = str(
            det.get("class_name") or det.get("label") or det.get("classification") or ""
        ).lower()
        if not any(keyword in name for keyword in DIRTY_KEYWORDS):
            continue
        prob = _normalize_probability(
            det.get("score") or det.get("confidence") or det.get("probability")
        )
        if prob is None:
            continue
        best = max(best, prob)
    return max(0.0, min(1.0, best))


def _score_temporal(metrics_by_frame: List[Dict[str, float]]) -> float:
    if not metrics_by_frame:
        return 55.0
    turbidity_values = [
        frame.get("turbidity_proxy")
        for frame in metrics_by_frame
        if frame.get("turbidity_proxy") is not None
    ]
    if len(turbidity_values) < 2:
        return 60.0
    mean = sum(turbidity_values) / len(turbidity_values)
    variance = sum((val - mean) ** 2 for val in turbidity_values) / len(
        turbidity_values
    )
    stability = max(0.0, 100.0 - min(variance * 2.0, 100.0))
    return _clamp(stability)


def _score_corroboration(
    media_count: int, variant_count: int, frames: int
) -> Tuple[float, float]:
    corroboration = min(media_count * 10.0 + frames * 2.0 + variant_count * 3.0, 100.0)
    confidence = min(media_count * 12.0 + frames * 1.5, 100.0)
    return _clamp(corroboration), _clamp(confidence)


def compute_scores(context: Dict[str, Any]) -> Dict[str, Any]:
    visual_avg: Dict[str, float] = context.get("visual_avg", {})
    detections: List[Dict[str, Any]] = context.get("detections", [])
    top_detection: Dict[str, Any] = context.get("top_detection", {}) or {}
    metrics_by_frame: List[Dict[str, float]] = context.get("metrics_by_frame", [])
    external: Dict[str, Any] | None = context.get("external")
    user_text: str | None = context.get("user_text")
    media_info: Dict[str, int] = context.get("media_info", {})

    user_assessment = context.get("user_text_assessment")
    classification_summary = context.get("classification_summary") or {}
    dirty_confidence = float(classification_summary.get("dirty_confidence") or 0.0)
    clean_confidence = float(classification_summary.get("clean_confidence") or 0.0)
    classification_available = bool(classification_summary.get("available"))

    external_score, external_confidence = _score_external(external)
    visual_score, image_quality_score = _score_visual_metrics(visual_avg)
    color_score = _score_color(visual_avg)
    model_confidence_score = _score_model_confidence(detections)
    temporal_score = _score_temporal(metrics_by_frame)

    classification_score = 50.0
    classification_confidence = 0.0
    if classification_available:
        classification_confidence = max(dirty_confidence, clean_confidence) * 100.0
        if dirty_confidence >= clean_confidence:
            classification_score = max(0.0, (1.0 - dirty_confidence) * 40.0)
            model_confidence_score = min(model_confidence_score, classification_score)
        else:
            classification_score = min(100.0, 60.0 + clean_confidence * 40.0)
            model_confidence_score = max(model_confidence_score, classification_score)

    detection_dirty_signal = _dirty_signal_from_detections(detections)
    top_name = str(
        top_detection.get("class_name")
        or top_detection.get("label")
        or top_detection.get("classification")
        or ""
    ).lower()
    if any(keyword in top_name for keyword in DIRTY_KEYWORDS):
        top_detection_prob = _normalize_probability(top_detection.get("score"))
        if top_detection_prob is not None:
            detection_dirty_signal = max(detection_dirty_signal, top_detection_prob)

    dirty_signal = max(dirty_confidence, detection_dirty_signal)

    if dirty_signal > 0.0:
        suppression = max(0.0, 1.0 - dirty_signal)
        visual_cap = suppression * 50.0
        color_cap = suppression * 60.0
        visual_score = min(visual_score, _clamp(visual_cap))
        color_score = min(color_score, _clamp(color_cap))
        if dirty_signal >= 0.6:
            temporal_score = min(temporal_score, _clamp(suppression * 70.0))

    user_text_active = False
    if isinstance(user_assessment, dict):
        if user_assessment.get("available"):
            user_text_score = _clamp(float(user_assessment.get("score", 50.0)))
            user_text_confidence = _clamp(
                float(user_assessment.get("confidence", 50.0))
            )
            user_text_active = True
        else:
            user_text_score, user_text_confidence = _score_user_text(user_text)
    elif user_text:
        user_text_score, user_text_confidence = _score_user_text(user_text)
        user_text_active = True
    else:
        user_text_score, user_text_confidence = _score_user_text(None)
    corroboration_score, corroboration_confidence = _score_corroboration(
        media_info.get("media_count", 0),
        media_info.get("variant_count", 0),
        media_info.get("frame_count", 0),
    )

    potability_weights = _normalize_weights(
        DEFAULT_POTABILITY_WEIGHTS,
        [
            "external" if external_score > 0 else None,
            "visual",
            "model_confidence",
            "color",
            "user_text" if user_text_active else None,
            "temporal",
            "corroboration",
            "custom_classification" if classification_available else None,
        ],
    )

    potability_components = {
        "external": ComponentScore(
            external_score, potability_weights.get("external", 0.0)
        ),
        "visual": ComponentScore(visual_score, potability_weights.get("visual", 0.0)),
        "model_confidence": ComponentScore(
            model_confidence_score, potability_weights.get("model_confidence", 0.0)
        ),
        "color": ComponentScore(color_score, potability_weights.get("color", 0.0)),
        "user_text": ComponentScore(
            user_text_score, potability_weights.get("user_text", 0.0)
        ),
        "temporal": ComponentScore(
            temporal_score, potability_weights.get("temporal", 0.0)
        ),
        "corroboration": ComponentScore(
            corroboration_score, potability_weights.get("corroboration", 0.0)
        ),
        "custom_classification": ComponentScore(
            classification_score, potability_weights.get("custom_classification", 0.0)
        ),
    }

    potability_score = sum(
        component.weighted() for component in potability_components.values()
    )
    potability_score = _clamp(potability_score)
    band_label = _band_from_score(potability_score)

    if dirty_signal >= 0.6:
        potability_score = min(potability_score, _clamp((1.0 - dirty_signal) * 35.0))
        band_label = _band_from_score(potability_score)

    confidence_weights = _normalize_weights(
        DEFAULT_CONFIDENCE_WEIGHTS,
        [
            "detector_confidence",
            "image_quality",
            "media_corroboration",
            "external" if external_confidence > 0 else None,
            "user_text" if user_text_active else None,
            "custom_classification" if classification_available else None,
        ],
    )

    confidence_components = {
        "detector_confidence": ComponentScore(
            model_confidence_score, confidence_weights.get("detector_confidence", 0.0)
        ),
        "image_quality": ComponentScore(
            image_quality_score, confidence_weights.get("image_quality", 0.0)
        ),
        "media_corroboration": ComponentScore(
            corroboration_confidence, confidence_weights.get("media_corroboration", 0.0)
        ),
        "external": ComponentScore(
            external_confidence, confidence_weights.get("external", 0.0)
        ),
        "user_text": ComponentScore(
            user_text_confidence, confidence_weights.get("user_text", 0.0)
        ),
        "custom_classification": ComponentScore(
            classification_confidence,
            confidence_weights.get("custom_classification", 0.0),
        ),
    }

    confidence_score = sum(
        component.weighted() for component in confidence_components.values()
    )
    confidence_score = _clamp(confidence_score)

    agreement_votes = 0
    if dirty_confidence >= 0.6:
        agreement_votes += 1
    if detection_dirty_signal >= 0.6:
        agreement_votes += 1
    if user_text_active and user_text_score <= 40.0:
        agreement_votes += 1
    if agreement_votes >= 2:
        confidence_score = max(confidence_score, 60.0 + agreement_votes * 10.0)
        confidence_score = _clamp(confidence_score)

    classification_payload: Dict[str, Any] = classification_summary.copy()
    classification_payload.setdefault("classification_score", classification_score)
    classification_payload.setdefault(
        "classification_confidence", classification_confidence
    )

    return {
        "potability_score": potability_score,
        "confidence_score": confidence_score,
        "band_label": band_label,
        "components": {
            "potability": {
                key: {"value": comp.value, "weight": comp.weight}
                for key, comp in potability_components.items()
            },
            "confidence": {
                key: {"value": comp.value, "weight": comp.weight}
                for key, comp in confidence_components.items()
            },
        },
        "classification_summary": classification_payload,
        "signals": {
            "dirty_signal": dirty_signal,
            "detection_dirty_signal": detection_dirty_signal,
            "dirty_confidence": dirty_confidence,
            "confidence_agreement_votes": agreement_votes,
        },
    }
