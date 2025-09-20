import os
from typing import Literal

DetectorBackend = Literal["rfdetr", "yolo11", "yolo12"]


class Settings:
    api_host: str = os.getenv("HYDROSCAN_API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("HYDROSCAN_API_PORT", "8001"))
    ui_port: int = int(os.getenv("HYDROSCAN_UI_PORT", "7860"))
    database_url: str = os.getenv(
        "HYDROSCAN_DB", "sqlite+aiosqlite:///./data/hydroscan.db"
    )
    media_dir: str = os.getenv("HYDROSCAN_MEDIA_DIR", "./data/media")
    results_dir: str = os.getenv("HYDROSCAN_RESULTS_DIR", "./data/results")
    cache_dir: str = os.getenv("HYDROSCAN_CACHE_DIR", "./data/cache")

    detector_backend: DetectorBackend = os.getenv(
        "HYDROSCAN_DETECTOR_BACKEND", "yolo11"
    )  # default YOLO11 per user
    detector_model: str | None = os.getenv("HYDROSCAN_DETECTOR_MODEL")
    detector_device: str = os.getenv("HYDROSCAN_DETECTOR_DEVICE", "cpu")

    ml_skip_download: bool = (
        os.getenv("HYDROSCAN_ML_SKIP_DOWNLOAD", "false").lower() == "true"
    )

    brand_websearch: bool = (
        os.getenv("HYDROSCAN_BRAND_WEBSEARCH", "false").lower() == "true"
    )
    brand_scraping: bool = (
        os.getenv("HYDROSCAN_BRAND_SCRAPING", "false").lower() == "true"
    )
    brand_rate_limit: int = int(os.getenv("HYDROSCAN_BRAND_RATE_LIMIT", "10"))
    brand_providers: list[str] = os.getenv(
        "HYDROSCAN_BRAND_PROVIDERS", "openfoodfacts,opencorporates"
    ).split(",")

    cache_ttl_s: int = int(os.getenv("HYDROSCAN_CACHE_TTL_S", "86400"))
    location_radius_km: float = float(os.getenv("HYDROSCAN_LOCATION_RADIUS_KM", "10.0"))


settings = Settings()
