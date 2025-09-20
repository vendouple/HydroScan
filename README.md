# HydroScan

Unified Python app: FastAPI backend + Gradio web UI. Upload images/videos and get a scene classification and a prototype potability score.

## Run (local dev)

1. Create a virtual environment and install dependencies from `requirements.txt`.
2. Launch the UI: run `python WebInterface/app.py`. The backend will autostart.

UI: http://127.0.0.1:7860 — Backend: http://127.0.0.1:8001

## Environment

- HYDROSCAN_DETECTOR_BACKEND: yolo11 | yolo12 (default yolo11)
- HYDROSCAN_DETECTOR_MODEL: path or alias like "yolo11n.pt"
- HYDROSCAN_DETECTOR_DEVICE: cpu | cuda | cuda:0
- HYDROSCAN_ML_SKIP_DOWNLOAD: true to skip Places365 downloads

Licensing notes: Ultralytics (YOLO11) is AGPL-3.0 or Enterprise. Places365 models under CC BY.
