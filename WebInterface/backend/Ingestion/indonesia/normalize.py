from __future__ import annotations

import datetime as dt
from typing import Any, Dict


PARAM_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "ph": {"min": 6.5, "max": 8.5},
    "turbidity_ntu": {"max": 5.0},
    "bod_mg_l": {"max": 3.0},
    "cod_mg_l": {"max": 25.0},
    "tds_mg_l": {"max": 500.0},
    "coli_mpn_100ml": {"max": 100.0},
}


def _score_parameter(name: str, value: float) -> float:
    info = PARAM_THRESHOLDS.get(name)
    if not info:
        return 60.0

    if "min" in info and "max" in info:
        if info["min"] <= value <= info["max"]:
            return 95.0
        deviation = min(abs(value - info["min"]), abs(value - info["max"]))
        penalty = min(deviation * 15.0, 90.0)
        return max(5.0, 95.0 - penalty)

    max_value = info.get("max")
    if max_value is not None:
        if value <= max_value:
            return 90.0
        overage_ratio = (value - max_value) / max_value
        penalty = min(overage_ratio * 120.0, 90.0)
        return max(5.0, 90.0 - penalty)

    return 60.0


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def normalize_external(raw: Dict[str, Any] | None) -> Dict[str, Any]:
    if not raw:
        return {}

    parameters_raw: Dict[str, Any] = raw.get("parameters", {})
    parameters: Dict[str, Dict[str, float]] = {}
    scores: list[float] = []

    for name, value in parameters_raw.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        score = _score_parameter(name, numeric_value)
        parameters[name] = {"value": numeric_value, "score": score}
        scores.append(score)

    if not scores:
        overall_quality = 0.0
    else:
        overall_quality = sum(scores) / len(scores)

    sample_date = _parse_date(raw.get("sample_date"))
    today = dt.date.today()
    recency_days = None
    if sample_date:
        recency_days = (today - sample_date).days

    return {
        "station_id": raw.get("station_id"),
        "station_name": raw.get("station_name"),
        "latitude": raw.get("latitude"),
        "longitude": raw.get("longitude"),
        "distance_km": raw.get("distance_km"),
        "sample_date": (
            sample_date.isoformat() if sample_date else raw.get("sample_date")
        ),
        "recency_days": recency_days,
        "parameters": parameters,
        "overall_quality": overall_quality,
        "source": raw.get("source"),
    }
