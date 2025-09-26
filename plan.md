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
- Automatic frame/image de-duplication: the system skips near-duplicate frames and near-duplicate still images to reduce processing time using adaptive change detection (cap 250 frames per minute of video, stop when frames are redundant).
- Internet access is allowed for model downloads and external public datasets.
- Persistence is enabled by default; analyses and media are stored under `history/`.
- Single entrypoint is `WebInterface/app.py` (runs backend and serves the frontend).
- Default repository branch is `master`.

---

## Backend implementation status (2025-09-25)

- Phase B (adapters & processing) completed: Places365, RF-DETR, and in-house model adapters implemented with unified outputs; filters, visual metrics, and aggregation pipeline wired into `/api/analyze`.
- Phase C scoring logic in place: potability/confidence calculator follows WHO-aligned weights with automatic normalization when external data is missing; Indonesian ingestion module fetches/caches nearest station data and normalizes parameter scores.
- Analyze flow now executes full pipeline (media validation, variant generation, detection, scoring, persistence) and stores structured results under `history/{analysis_id}/` along with optional debug artifacts.
- `/api/models/status` exposes asset health reporting sha256 hashes for downloaded checkpoints.

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
│     │  ├─ InHouse/
│     │  │  ├─ CLS.pt             # in-house fine-tuned YOLOv11 classification model (clean vs dirty)
│     │  │  └─ OBB.pt             # in-house fine-tuned YOLOv11 object detection model (oriented bounding boxes)
│     │  ├─ CustomModel/           # legacy model location (for backward compatibility)
│     │  │  ├─ Classification.pt   # alternative classification weights
│     │  │  └─ ObjectDetection.pt  # alternative object detection weights
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

- The timeline acts like a structured debug log. Each entry mirrors a pipeline step with granular detail text supplied by the backend (filters chosen, detector scores, branch decisions, etc.).
- Core steps (always shown in skeleton):
  1. `Media ingestion` → validates uploads, extracts frames, performs similarity-based deduplication (cap 250 extracted frames per minute of video).
  2. `Scene detection (Places365)` → reports dominant indoor/outdoor label and confidence.
  3. `Scene routing` → states whether outdoor path (external data) or indoor/unknown path (packaging & OCR) is engaged.
  4. `Adaptive filter stack` → lists filters applied (e.g., "White balance", "Contrast stretch", "Highlight suppression").
  5. `Detection engines` → detector confidence summary, fallback usage, top detections.
  6. `Aggregation` → cross-variant fusion outcome (detection count, top class/score).
  7. `Water confirmation` → consolidated verdict (outdoor cues, packaging crops, or warnings).
  8. `Scoring` → band label and overall status.
  9. `Finalizing` → persistence vs. ephemeral result.
- Optional entries appear when relevant and are logged inline with statuses:
  - `Outdoor external data` & `Outdoor detector` (location-based fetch, fallback to in-house model when station data missing).
  - `Packaging detection`, `Brand & OCR`, `Liquid confirmation` for indoor/unknown scenes.
  - `User notes analysis` when the LLaMA adapter interprets text.
- Each entry surfaces status icons (pending / in-progress / done / warning / error) and timestamp/description so operators can treat the timeline as a readable audit trail.

**Debug mode**

- Checkbox `Debug` in UI. If checked, the timeline area stays verbose and the debug console streams backend log lines (adapter selection, filter boosts, external data issues, fallback decisions).
- Detection snapshots are grouped by source in collapsible sections (e.g., `Scene classifier`, `Adaptive filters`, `RF-DETR`, `OCR & crops`). Each filtered variant snapshot should use human-readable filter names (e.g., "White balance", "Highlight suppression").
- Debug pane supports copying logs and downloading debug artifacts as JSON/ZIP, while image snapshots remain inline for rapid inspection.
- Dedicated **Model Lab** page allows operators to choose an adapter (Places365, RF-DETR, In-house YOLO) and run isolated inference on a single image. Results show raw outputs, annotated previews, and—on failure—tracebacks and diagnostic notes so issues can be reproduced quickly.

**Results area**

- Show `Potability Score` (large number 0–100) and `Band Label`.
- Show `Confidence Score` (0–100) and a colored badge: High / Moderate / Low.
- Show `Cleanliness levels` (visual breakdown): `Turbidity proxy`, `Color deviation`, `Visible particulates` with small bar/score each.
- Show provenance: which filter & frame produced the top detection and surface thumbnails (annotated) when available.
- Expose history affordance: analyses are persisted by default but can be skipped via "Don’t save this" checkbox. When saved, the “Recent analyses” panel can reload past results and their debug artifacts.

**Interaction & experience**

- When Analyze is pressed, the UI adds a top-level `analysis-mode` class so layout adapts: timeline becomes visible, results area prepares streaming updates, upload control disabled until finished.
- Users can opt out of persistence per run (`dont_save` checkbox). When disabled, debug artifact download links are hidden/disabled.
- Provide progress indicators for long tasks; provide link to continue polling or to retrieve results later (analysis ID).
- History list surfaces saved analyses with timestamps and quick links to reload results/debug snapshots.

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
  - Load two separate fine-tuned YOLOv11 models from `backend/Models/InHouse/`:
    - `CLS.pt`: Water quality classification model ("bersih" vs "kotor")
    - `OBB.pt`: Object detection model with oriented bounding boxes
  - Provide `classify_water(image)` for water quality classification
  - Provide `detect_objects(image)` for object detection with bounding boxes
  - Provide `predict_comprehensive(image)` combining both models for complete analysis
  - Work with filter algorithms to reach confidence thresholds through adaptive processing
  - Support confidence-driven filter application to enhance detection accuracy

---

# 5. Processing pipeline (detailed)

1. **Ingestion & validation**

   - Validate counts (<=25 images, <=5 videos) and video lengths (<=60s). Reject or truncate if policy violated.
   - Extract frames from videos using adaptive similarity-based sampling (see below).

2. **Frame sampling**

   - For each video, extract frames using similarity detection to avoid redundant processing:
     - Compare consecutive frames using structural similarity (SSIM) or histogram difference
     - Skip frames that are too similar to previously selected frames (threshold: 18.0 difference)
     - Maximum 250 frames per video regardless of length
     - Prefer frames with good exposure and minimal motion blur
   - This reduces processing time while maintaining analysis quality

3. **Variants generation (filters)**

- For each frame produce adaptive variants using confidence-driven filter selection:
  - **Base filters**: `Original`, `White balance`, `Contrast stretch`
  - **Enhancement filters**: `Denoise`, `Edge sharpen`, `Gamma boost`, `Highlight suppression`
  - **Color filters**: `Red enhancement`, `Blue enhancement`, `Green enhancement`, `Cyan filter`, `Yellow filter`, `Magenta filter`
- Apply filters progressively until detector confidence meets target (default 0.6) or maximum filters reached
- Compare filter results and select best-performing variants based on detection confidence
- Map internal operation keys to user-friendly names for debug snapshots (e.g., `gamma_1.2` → "Gamma boost 1.2", `red_enhance` → "Red enhancement")

4. **Scene classification**

   - Run Places365 on a representative set of frames (e.g., 3 frames sampled across the media). Compute majority or weighted scene label.

5. **Scene routing & external data**

- After Places365 majority vote, route pipeline:
  - **Outdoor** path → attempt location-based water quality fetch (within configurable radius). If location missing or station unavailable, fall back to in-house detector.
  - **Indoor/Unknown** path → engage packaging detection and OCR to infer brand. Check if brand is water-related:
    - If recognizable water brand → fetch brand quality data if available
    - If non-water brand → reject with error
    - If brand unknown → crop to water area in packaging (if transparent liquid visible) → use in-house model
    - If no packaging/liquid visible → end analysis (water not detected)
- Outdoor path still triggers ingestion `backend/Ingestion/indonesia` when `lat/lon` supplied.
- **Adaptive filter strategy**: Use confidence-based filter selection. If detector confidence is low, progressively apply more filters (original → white balance → contrast → denoise → color enhancements) until confidence target is reached or all filters exhausted.

6. **Detection**

- Run InModel adapter with both fine-tuned YOLOv11 models (CLS.pt + OBB.pt) on adaptive variants.
- CLS.pt provides water quality classification (clean vs dirty) with confidence scores.
- OBB.pt provides object detection with oriented bounding boxes for water quality indicators.
- Adaptive filter application continues until target confidence threshold is reached or all filters exhausted.
- Fall back to RF-DETR when InModel confidence is insufficient or models unavailable.
- Track objects across frames via IoU-based tracker to build stability and persistence evidence.
- Store annotated snapshots per detector/variant for debugging.

7. **Visual metrics**

   - Compute turbidity proxy (particle counts, backscatter proxy), color distance (HSV or CIE metric), foam / scum detectors, edge density.

8. **Aggregation**

   - For each metric, produce a normalized 0–100 score.
   - Keep provenance: which frame & filter produced each top metric.

9. **Scoring**

- Call scoring engine (see §6) to compute `potability_score` and `confidence_score`. Link final band label back to timeline entry (`Scoring`).

10. **Return results**

- Return summary, components, timeline (detailed log), and (if debug) a rich `debug` object with grouped snapshots (`scene`, `filters`, `detector`, `ocr`) plus manifest metadata.
- Include detector error diagnostics and stack traces when a model invocation fails so downstream consumers can inspect root causes.

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
- `POST /api/debug/run` — run a single adapter against an uploaded image (`model={place365|rf_detr|inmodel}`) and return structured results, annotated previews, and stack traces when errors occur

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
- Capture per-filter detector failures (message + traceback) and surface them both in timeline warnings and in the returned `debug.errors` manifest.
- Provide Model Lab interface (see §3) for manual adapter verification with transparent error reporting.

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
