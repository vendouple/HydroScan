# HydroScan

Prototype to analyze water photos/videos and compute Potability and Confidence scores. Single start command serves a web UI and API.

- Entry point: `python WebInterface/app.py`
- Health check: `GET /health`
- UI: `GET /`

## Setup

- Python 3.x
- Install dependencies:

```
python -m pip install -r requirements.txt
```

## Notes

- On first run, the app may attempt to fetch Places365 assets. If automatic download fails, place files manually:
  - `WebInterface/backend/Models/Place365/deploy_resnet152_places365.prototxt`
  - `WebInterface/backend/Models/Place365/resnet152_places365.caffemodel`
- History is persisted under `history/` by default.

## Development

- Routers live in `WebInterface/API/`
- Backend pipeline lives in `WebInterface/backend/`
- Frontend assets in `WebInterface/frontend/`
