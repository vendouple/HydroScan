# plan.md — HydroScan (Final Development-Ready)

**Branch**: `master` | **Date**: 2025-09-22

**Primary goal**: Unified Python prototype that analyzes photos & short videos of water, uses visual + external data to compute a **Potability Score (0–100%)** and a **Confidence Score (0–100%)**. Prioritizes Indonesian external data for verification. Local-first inference (Places365 for scene classification; RF-DETR / YOLOv11 / YOLOv12 family for object detection). Single startup: `WebInterface/app.py` boots backend services and serves the frontend.

---

# 0. Executive summary

- Input: up to **25 images** and/or up to **5 videos** (each ≤ 60s) + a **free-text paragraph** describing smell/color/feel/temperature.
- Output: `potability_score` (0–100%), `confidence_score` (0–100%), `band_label` (Drinkable / Very Clean / Clean for daily use / Less clean / Unclean), and detailed debug artifacts on demand.
- Standard: **WHO Guidelines for Drinking-Water Quality (4th edition)** as reference thresholds.
- Priority region: **Indonesia** (data ingestion / mapping first); other regions are placeholders.
- Models: local-only run; auto-download certain external model assets (Places365 prototxt & caffemodel; optional RF-DETR/Yolo weights). Internet access is allowed for public model/data downloads.
- UI: glassmorphism HTML/CSS frontend (clean/adaptive while analyzing) with a processing **timeline/status area** that shows current filters, processing steps, and (if debug enabled) a console-like log output.
- Single command to start: `python WebInterface/app.py`.

---

Clarifications applied (2025-09-22):

- Python 3.x; CUDA multi-GPU supported with CPU fallback.
- Primary object detector is RF-DETR. In-house YOLOv11/YOLOv12-compatible models handle custom classes (algae, twigs, particulates) for detection and classification.
- Automatic frame/image de-duplication: the system skips near-duplicate frames and near-duplicate still images to reduce processing time.
- Internet access is allowed for model downloads and external public datasets.
- Persistence is enabled by default; analyses and media are stored under `history/`.
- Single entrypoint is `WebInterface/app.py` (runs backend and serves the frontend).
- Default repository branch is `master`.

---

# 1. Required model links (Places365 exact assets)

Use these exact resources for Places365 auto-download (required):

- **Prototxt (deploy ResNet152 Places365)** (needed):
  `https://raw.githubusercontent.com/CSAILVision/places365/master/deploy_resnet152_places365.prototxt`

- **Caffe weights (ResNet152 Places365)** (needed):
  `http://places2.csail.mit.edu/models_places365/resnet152_places365.caffemodel`

> These are required (Caffe prototxt + caffemodel) when running Places365 in a Caffe runtime or for conversion to PyTorch/ONNX before use.

---

# 2. Repository layout (final, exact)

```
/ (repo root)
├─ WebInterface/
│  ├─ app.py                      # single entrypoint: boot backend + serve frontend
│  ├─ __init__.py
│  ├─ frontend/
│  │  ├─ templates/
│  │  │  └─ index.html
│  │  └─ static/
│  │     ├─ css/main.css
│  │     └─ js/ui.js
│  ├─ API/                        # endpoints to talk to frontend
│  │  ├─ __init__.py
│  │  ├─ analyze.py
│  │  └─ results.py
│  └─ backend/
│     ├─ __init__.py
│     ├─ Models/
│     │  ├─ CustomModel/
│     │  │  ├─ Classification.pt   # in-house classification weights (yolo11/yolo12 compatible)
│     │  │  └─ ObjectDetection.pt  # in-house object bounding box weights
│     │  ├─ Place365/
│     │  │  ├─ deploy_resnet152_places365.prototxt
│     │  │  └─ resnet152_places365.caffemodel
│     │  └─ ObjectDetection/       # external detector checkpoints (rf-detr, ultralytics, etc.)
│     │     ├─ rf_detr_checkpoint.pth
│     │     ├─ yolo11n.pt
│     │     └─ yolov12_best.pt
│     ├─ Adapters/
│     │  ├─ __init__.py
│     │  ├─ Place365.py            # adapter to run Places365
│     │  ├─ RFDETR.py              # adapter to run RF-DETR (example usage included)
│     │  └─ InModel.py             # adapter to run in-house YOLOv11/YOLOv12 models
│     ├─ Processing/
│     │  ├─ __init__.py
│     │  ├─ filters.py             # image/video filter pipeline
│     │  ├─ visual_metrics.py      # turbidity proxies, color distance, foam detection
│     │  └─ aggregator.py          # aggregate detections across variants/frames
│     ├─ Scoring/
│     │  ├─ __init__.py
│     │  └─ scoring.py             # potability score + confidence calculation
│     ├─ Ingestion/
│     │  ├─ __init__.py
│     │  └─ indonesia/             # data.go.id, KLHK, PUPR ingestion + normalizers
│     │     ├─ __init__.py
│     │     ├─ fetch_data.py
│     │     └─ normalize.py
│     ├─ Workers/                  # worker pool, parallel filter inference
│     │  └─ __init__.py
│     └─ Utils/
│        ├─ __init__.py
│        └─ fetch_models.py        # auto-download + checksum utility (invoked by app.py)
├─ history/                        # persisted analyses (enabled by default)
│  └─ .gitkeep
├─ requirements.txt
└─ README.md
```

Notes:

- `WebInterface/app.py` is the single entrypoint. At startup, it runs `WebInterface/backend/Utils/fetch_models.py` (if models are missing), loads adapters, starts the HTTP server (FastAPI/Starlette) and serves static templates.
- `WebInterface/backend/Models/Place365` must contain the prototxt and caffemodel as filenames listed above.

---

# 3. Frontend requirements (glassmorphism + timeline + debug)

**General look & feel**

- UI uses glassmorphism (translucent cards, subtle borders, soft shadows, gradient accent).
- Minimal, mobile-first responsive layout.

**Uploader & input**

- Files input (images & videos), plus a single **paragraph textbox** for user description (smell, color, feel, temperature).
- Buttons: `Analyze`, `Clear`, optional `Allow location` for external data.

**Processing status area (timeline)**

- A left or right column shows a **timeline / status area** that lists processing stages in order and updates in real time (or via polling):
  1. "Preparing media" (extract frames)
  2. "Applying filters" → shows which filter(s) are running (e.g., `original`, `denoise`, `contrast_stretch`)
  3. "Running scene classifier (Places365)" → show `scene: outdoor (0.92)`
  4. "Fetching external data" → show source & station (if applicable)
  5. "Running detector (RF-DETR / YOLO)" → per-frame progress and top detection(s)
  6. "Aggregating results" → computing scores
  7. "Finalizing" → saving debug artifacts
- Each timeline item has a status icon (pending / in-progress / done / error) and optional timestamp.

**Debug mode**

- Checkbox `Debug` in UI. If checked, the timeline area also includes a small console-like pane showing `console.log` messages from the backend (streamed or polled). This includes adapter loading logs, filter choices, model confidences, and warnings.
- Debug pane supports copying logs and downloading debug artifacts as a ZIP.

**Results area**

- Show `Potability Score` (large number 0–100) and `Band Label`.
- Show `Confidence Score` (0–100) and a colored badge: High / Moderate / Low.
- Show `Cleanliness levels` (visual breakdown): `Turbidity proxy`, `Color deviation`, `Visible particulates` with small bar/score each.
- Show `Which filter produced best detection` and a thumbnail of the best frame + annotated boxes.

**Interaction & experience**

- When Analyze is pressed, the UI adds a top-level `analysis-mode` class so layout adapts: timeline becomes visible, results area prepares streaming updates, upload control disabled until finished.
- Provide progress indicators for long tasks; provide link to continue polling or to retrieve results later (analysis ID).

---

# 4. Adapters (detailed)

**Adapter goals**: present a unified, stable API to the processing pipeline regardless of underlying model.

**All adapters must return standard Python data structures and not leak backend internals.**

## 4.1 `backend/Adapters/Place365.py` (adapter)

- Responsibilities:
  - Load Places365 model (Caffe prototxt + caffemodel) or converted PyTorch model.
  - Expose `classify(pil_image)` → `{ "scene": "outdoor"|"indoor"|"unknown", "label": "park", "confidence": 0.92 }`.
  - Provide fallback path if Caffe isn't available: attempt ONNX or PyTorch conversion (scripts provided separately) and a clear error message.

## 4.2 `backend/Adapters/RFDETR.py` (adapter)

- Responsibilities:
  - Wrap `RFDETRBase` usage as you supplied (optimize_for_inference, predict).
  - Expose `predict(pil_image, threshold=0.5)` → list of unified `Detection` dicts:
    ```py
    {
      "bbox": [x1,y1,x2,y2],
      "class_id": int,
      "class_name": str,
      "score": float,
      "source": "rfdetr"
    }
    ```
  - Provide batch/pipelined inference functions for multiple images.

## 4.3 `backend/Adapters/InModel.py` (adapter)

- Responsibilities:
  - Load in-house models (YOLOv11 / YOLOv12) from `backend/Models/CustomModel/*.pt`.
  - Provide `predict(images, conf=0.25, iou=0.45)` returning unified detections.
  - Support model selection via env var: `MODEL_BACKEND=inmodel|rfdetr`.

---

# 5. Processing pipeline (detailed)

1. **Ingestion & validation**

   - Validate counts (<=25 images, <=5 videos) and video lengths (<=60s). Reject or truncate if policy violated.
   - Extract frames from videos using adaptive sampling (see below).

2. **Frame sampling**

   - For each video, sample up to `MAX_FRAMES_PER_VIDEO` frames using a strategy: uniform sampling capped at e.g. 60 frames per video or lower depending on CPU budget.
   - Prefer keyframes or choose frames with minimum motion blur where possible.

3. **Variants generation (filters)**

   - For each frame produce variants (configurable):
     - `original`
     - `auto_white_balance`
     - `contrast_stretch` (CLAHE)
     - `denoise` (non-local means)
     - `sharpen`
     - `gamma_correction` (gamma variations)
     - `deglare` / highlight suppression
   - Generate 3–6 variants per frame by default.

4. **Scene classification**

   - Run Places365 on a representative set of frames (e.g., 3 frames sampled across the media). Compute majority or weighted scene label.

5. **External data fetch**

   - If scene `outdoor` and `lat/lon` provided and permitted by user, run ingestion `backend/Ingestion/indonesia` to fetch station data. Choose nearest station within `EXTERNAL_MAX_DISTANCE_KM` (default 25 km).

6. **Detection**

   - Run detector adapter (RF-DETR primary, InModel optional) on all variants and aggregate.
   - Track objects across frames via IoU-based tracker to build stability and persistence evidence.

7. **Visual metrics**

   - Compute turbidity proxy (particle counts, backscatter proxy), color distance (HSV or CIE metric), foam / scum detectors, edge density.

8. **Aggregation**

   - For each metric, produce a normalized 0–100 score.
   - Keep provenance: which frame & filter produced each top metric.

9. **Scoring**

   - Call scoring engine (see §6) to compute `potability_score` and `confidence_score`.

10. **Return results**
    - Return summary, components, and (if debug) a rich `debug` object with annotated images, selected frames, filter provenance, and logs.

---

# 6. Scoring: Potability & Confidence (final)

### 6.1 Score bands (as clarified)

- **100%**: Drinkable — **must be corroborated by external data** (recent WHO-type data for the same location + model confidence).
- **99–51%**: Very clean, but not drinkable.
- **50%**: Clean for daily use, not drinkable.
- **49–26%**: Less clean — avoid daily use.
- **<25%**: Unclean — unsafe.

### 6.2 Default component weights (sum = 100)

|                                Component | Default weight |
| ---------------------------------------: | -------------: |
|    External WHO-comparable data presence |             30 |
|         Visual clarity & turbidity proxy |             20 |
| Model detection confidence & consistency |             15 |
|                 Visible color/appearance |             10 |
|                    User free-text signal |             10 |
|                 Video temporal stability |             10 |
|                Media corroboration count |              5 |

> Re-normalization: if `external` data is missing, distribute its 30 weight proportionally to other components or re-normalize the remaining weights so they sum to 100 (configurable).

### 6.3 Confidence score components (0–100)

- Detector confidence mean (40%)
- Image/video quality (sharpness, exposure) (20%)
- Media corroboration count (15%)
- External data presence & recency (15%)
- User text concordance (10%)

**Human-readable bands**: >80 = High, 50–80 = Moderate, <50 = Low.

### 6.4 Mapping visual metrics

- Turbidity proxy → NTU-like mapping (calibrate with labeled dataset) → map to 0–100 quality.
- Color distance → CIEDE2000 or HSV distance mapping → 0–100.
- Temporal stability → low variance → increase score; high variance → penalize.

---

# 7. External Indonesia data (priority) — sources & how to integrate

**Targets & notes**

1. **data.go.id** — Indonesia open data portal. Search for "kualitas air" or "mutu air sungai". Download CSVs and ingest.

   - Approach: fetch CSV, normalize field names, units; convert units; store under `backend/Ingestion/indonesia/`.

2. **KLHK / PPKL** — monitoring dashboards (station-level time series). Inspect network XHR endpoints; if public, use them; else consider formal data request.

3. **PUPR / SIHKA** — hydrology & water quality portal. Fetch station lists & time series.

4. **WHO** — use GDWQ tables for thresholds. Keep a local copy of relevant parameter tables.

**Integration steps**

- Downloader: `backend/Ingestion/indonesia/fetch_data.py` → stores raw + normalized CSV/JSON.
- Normalizer: `backend/Ingestion/indonesia/normalize.py` → maps parameter names to canonical (ph, tds, turbidity, bod, cod, coli, heavy metals).
- Geo-matching: nearest station search using Haversine, configurable radius (default 25 km).
- Caching: cache responses for `EXTERNAL_TTL_SECONDS` (default 24h).
- Provenance: always return source name, sample date, and station id used for scoring.

---

# 8. Auto-download & boot sequence

**WebInterface/backend/Utils/fetch_models.py** must:

- Download Places365 prototxt + caffemodel (exact links in §1).
- Optionally download RF-DETR, yolo11n, yolov12 weights if configured.
- Verify checksums (SHA256) and fail gracefully (UI shows degraded mode) if network unavailable.

**Boot order (app.py)**:

1. run `WebInterface/backend/Utils/fetch_models.py` (skip if models exist & checksum ok)
2. initialize adapters (Place365, RFDETR, InModel)
3. start FastAPI server and static file server
4. expose `/api/models/status` for frontend health check

---

# 9. API endpoints (contract summary)

- `POST /api/analyze` — multipart form: `media[]`, `text_paragraph`, `lat`, `lon`, `debug` (bool)

  - returns: `analysis_id` (if queued) or immediate result JSON

- `GET /api/results/{analysis_id}` — returns stored result JSON

- `GET /api/models/status` — lists available models & checksums

**Response schema (summary)**

```json
{
  "analysis_id": "uuid",
  "potability_score": 62.4,
  "confidence_score": 71,
  "band_label": "Clean for daily use, not drinkable",
  "components": {
    "external": 12.0,
    "visual": 30.5,
    "model_confidence": 10.2,
    "user_input": 9.7
  },
  "debug": {
    /* optional artifacts */
  }
}
```

---

# 10. Debugging & auditability

- `Debug` checkbox enables detailed streaming logs to the frontend timeline console.
- Save debug artifacts (annotated frames, selected best variants, filter that yielded best detection) when `debug=true`.
- Allow download of debug ZIP via `GET /api/results/{id}/debug.zip`.

---

# 11. Security & privacy

- Display a prominent disclaimer: _HydroScan provides informational analysis only and is not a substitute for laboratory testing or public health advice._
- Do not persist media unless `HISTORY_ENABLED=true`. If persisted, encrypt at rest and provide deletion controls.
- Respect `robots.txt` for any scraping of agency dashboards; prefer public APIs/CSV exports.

---

# 12. Development checklist (actionable tasks)

**Phase A — scaffold & boot**

1. Create repo skeleton (exact layout above).
2. Implement `scripts/fetch_models.py` with Places365 links.
3. Implement `WebInterface/main.py` (bootstrap + start server).

**Phase B — adapters & processing** 4. Implement `backend/Adapters/Place365.py`. 5. Implement `backend/Adapters/RFDETR.py` (use provided example code as basis). 6. Implement `backend/Adapters/InModel.py` for YOLOv11/YOLOv12. 7. Implement `backend/Processing/filters.py` and `visual_metrics.py`. 8. Implement `backend/Processing/aggregator.py` to union detections across filters/frames.

**Phase C — scoring & external data** 9. Implement `backend/Scoring/scoring.py` (WHO thresholds mapping and weights; confidence calculator). 10. Implement `backend/Ingestion/indonesia` ingestion + normalizer + cache.

**Phase D — API & frontend** 11. Implement `WebInterface/API/analyze.py` and `results.py` with job queue support. 12. Implement frontend UI (glassmorphism) with timeline, debug console, streaming updates. 13. Integrate results & debug artifact download.

**Phase E — testing & calibration** 14. Collect Indonesia-labeled image + turbidity data for calibration. 15. Tune turbidity mapping and weights; add unit tests.

---

# 13. Example RF-DETR snippet (to include in adapter) — from your sample

```py
import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase

model = RFDETRBase()
model.optimize_for_inference()

url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"
image = Image.open(io.BytesIO(requests.get(url).content))
detections = model.predict(image, threshold=0.5)

labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)
```

Implement the above in `backend/Adapters/RFDETR.py` and return the standardized detection objects used in processing.

---

# 14. Example mini `app.py` (concept)

```py
# WebInterface/app.py
import os
from fastapi import FastAPI
from WebInterface.backend.Utils.fetch_models import ensure_models

# 1) ensure models
ensure_models(os.path.join(os.path.dirname(__file__), "backend", "Models"))

# 2) init adapters (example)
from WebInterface.backend.Adapters.Place365 import Place365Adapter
from WebInterface.backend.Adapters.RFDETR import RFDETRAdapter
from WebInterface.backend.Adapters.InModel import InModelAdapter

place_adapter = Place365Adapter(models_dir=os.path.join(os.path.dirname(__file__), "backend", "Models"))
rfdetr_adapter = RFDETRAdapter(checkpoint_path=os.path.join(os.path.dirname(__file__), "backend", "Models", "ObjectDetection", "rf_detr_checkpoint.pth"))
inmodel_adapter = InModelAdapter(weights_path=os.path.join(os.path.dirname(__file__), "backend", "Models", "CustomModel", "ObjectDetection.pt"))

# 3) create app and mount routers
app = FastAPI()
# register routers from WebInterface/API
```

---

# 15. Deliverables I can produce next (pick any)

- `scripts/fetch_models.py` file with exact Places365 links (I will compute checksums or you can supply them).
- `WebInterface/main.py` starter ready to run (FastAPI, static file mounting, health endpoints).
- `backend/Adapters/Place365.py`, `backend/Adapters/RFDETR.py`, `backend/Adapters/InModel.py` skeletons.
- Frontend files: `index.html`, `main.css`, `ui.js` ready to paste.
- `backend/Ingestion/indonesia/fetch_data.py` starter that crawls `data.go.id` CSVs.

Tell me which file(s) you want me to generate now and I will create them as ready-to-paste code blocks or save them in the workspace.
