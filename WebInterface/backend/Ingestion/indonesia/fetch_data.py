from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import requests  # type: ignore

    _HAS_REQUESTS = True
except Exception:  # pragma: no cover - optional dependency during offline dev
    requests = None  # type: ignore
    _HAS_REQUESTS = False


DATASET_URL = os.environ.get(
    "HYDROSCAN_INDONESIA_DATASET",
    "https://raw.githubusercontent.com/noneeeeeeeeeee/hydroscan-datasets/main/indonesia/water_quality_sample.csv",
)
CACHE_DIR = Path(os.environ.get("HYDROSCAN_CACHE_DIR", Path("history/.cache")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "indonesia_water_quality.json"
CACHE_TTL_SECONDS = int(os.environ.get("HYDROSCAN_EXTERNAL_TTL_SECONDS", "86400"))


@dataclass(slots=True)
class StationRecord:
    station_id: str
    name: str
    latitude: float
    longitude: float
    sample_date: str | None
    parameters: Dict[str, Any]
    source: str


FALLBACK_SAMPLE: List[StationRecord] = [
    StationRecord(
        station_id="sample-001",
        name="Ciliwung River - Bogor",
        latitude=-6.597146,
        longitude=106.806039,
        sample_date="2025-07-18",
        parameters={
            "ph": 7.1,
            "turbidity_ntu": 9.5,
            "bod_mg_l": 2.3,
            "cod_mg_l": 13.0,
            "tds_mg_l": 320,
            "coli_mpn_100ml": 120,
        },
        source="fallback-sample",
    ),
    StationRecord(
        station_id="sample-002",
        name="Cisadane River - Tangerang",
        latitude=-6.189428,
        longitude=106.707878,
        sample_date="2025-07-12",
        parameters={
            "ph": 6.8,
            "turbidity_ntu": 18.0,
            "bod_mg_l": 3.6,
            "cod_mg_l": 21.0,
            "tds_mg_l": 410,
            "coli_mpn_100ml": 260,
        },
        source="fallback-sample",
    ),
    StationRecord(
        station_id="sample-003",
        name="Brantas River - Surabaya",
        latitude=-7.257472,
        longitude=112.75209,
        sample_date="2025-06-30",
        parameters={
            "ph": 7.4,
            "turbidity_ntu": 12.0,
            "bod_mg_l": 2.9,
            "cod_mg_l": 16.0,
            "tds_mg_l": 360,
            "coli_mpn_100ml": 180,
        },
        source="fallback-sample",
    ),
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return float(R * c)


def _load_dataset_from_cache() -> Optional[List[StationRecord]]:
    if not CACHE_PATH.exists():
        return None
    try:
        if time.time() - CACHE_PATH.stat().st_mtime > CACHE_TTL_SECONDS:
            return None
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [StationRecord(**entry) for entry in data]
    except Exception:
        return None


def _save_dataset_to_cache(records: Iterable[StationRecord]) -> None:
    try:
        serializable = [record.__dict__ for record in records]
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_dataset_from_remote() -> Optional[List[StationRecord]]:
    if not _HAS_REQUESTS or not DATASET_URL:
        return None
    try:
        response = requests.get(DATASET_URL, timeout=(10, 30))
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        text = response.text
        records: List[StationRecord] = []
        if "json" in content_type:
            payload = response.json()
            for entry in payload:
                records.append(
                    StationRecord(
                        station_id=str(entry.get("station_id") or entry.get("id")),
                        name=entry.get("name")
                        or entry.get("station_name")
                        or "Unknown station",
                        latitude=float(entry.get("latitude")),
                        longitude=float(entry.get("longitude")),
                        sample_date=entry.get("sample_date"),
                        parameters=entry.get("parameters") or {},
                        source=entry.get("source") or DATASET_URL,
                    )
                )
        else:
            reader = csv.DictReader(text.splitlines())
            for row in reader:
                try:
                    parameters = {
                        key: float(row[key])
                        for key in row
                        if key
                        and key
                        not in {
                            "station_id",
                            "name",
                            "latitude",
                            "longitude",
                            "sample_date",
                            "source",
                        }
                        and row[key]
                    }
                    records.append(
                        StationRecord(
                            station_id=row.get("station_id")
                            or row.get("id")
                            or f"station-{len(records)+1}",
                            name=row.get("name")
                            or row.get("station_name")
                            or "Unknown station",
                            latitude=float(row["latitude"]),
                            longitude=float(row["longitude"]),
                            sample_date=row.get("sample_date") or row.get("date"),
                            parameters=parameters,
                            source=row.get("source") or DATASET_URL,
                        )
                    )
                except Exception:
                    continue
        if records:
            _save_dataset_to_cache(records)
            return records
    except Exception:
        return None
    return None


def _load_dataset() -> List[StationRecord]:
    cached = _load_dataset_from_cache()
    if cached:
        return cached
    remote = _load_dataset_from_remote()
    if remote:
        return remote
    return FALLBACK_SAMPLE


def fetch_nearest_station(
    lat: float, lon: float, max_km: float = 25.0
) -> Dict[str, Any] | None:
    dataset = _load_dataset()
    nearest: Optional[StationRecord] = None
    nearest_distance = float("inf")
    for record in dataset:
        distance = _haversine_km(lat, lon, record.latitude, record.longitude)
        if distance <= max_km and distance < nearest_distance:
            nearest = record
            nearest_distance = distance

    if nearest is None:
        return None

    return {
        "station_id": nearest.station_id,
        "station_name": nearest.name,
        "latitude": nearest.latitude,
        "longitude": nearest.longitude,
        "sample_date": nearest.sample_date,
        "parameters": nearest.parameters,
        "distance_km": nearest_distance,
        "source": nearest.source,
    }
