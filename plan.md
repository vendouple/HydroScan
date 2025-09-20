# Implementation Plan: Water Impurity Analysis from Images and Video

**Branch**: `main` | **Date**: 2025-09-19 | **Spec**: D:\dev\Github\HydroScan\specs\001-build-an-app\spec.md
**Input**: Feature specification from `D:\dev\Github\HydroScan\specs\001-build-an-app\spec.md`

## Execution Flow (/plan command scope)

```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `.github/copilot-instructions.md` for GitHub Copilot)
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

## Summary

Primary requirement: Enable users to analyze photos/videos for water impurities with Outdoor vs Indoor flows, optional location/brand/OCR lookups, potability scoring (0–100%), adaptive video sampling (≤60s), filter comparisons, and Debug Mode. We commit to a unified Python runtime where the Gradio UI autostarts the FastAPI backend. For scene classification we will use external pretrained Places365 (ResNet) models; for object detection we will use RF-DETR (Apache-2.0) as the baseline detector, with explicit compatibility for Ultralytics YOLO11 (aka "YOLOv11") and an in-house YOLOv12 (AGPL-3.0) training path.

## End-to-End Analysis Workflow (user-visible timeline)

The UI will present a simple step timeline by default and a detailed debug timeline when Debug Mode is enabled. High-level flow:

1. Scene Detection

- Classify scene: Outdoor/Natural vs Indoor/Unknown with confidence.
- If confidence low, allow user to confirm manually (one-tap toggle) and continue.

2. Scene Confirmed → Branching

- Outdoor/Natural:
  - Request location permission.
  - If granted: Retrieve external water-quality context near location; attribute sources; combine with media analysis.
  - If denied/unavailable: Continue with media-only analysis.
  - Fallbacks: If external data missing/ambiguous, proceed with in-house models using water-region analysis.
- Indoor/Unknown:
  - Run object detection for containers/labels/brands.
  - If bottle/brand detected: OCR label → attempt brand match → fetch brand water data; if none, fall back to media-only analysis.
  - If brand unknown: crop to water regions and run in-house models for impurity cues.
  - If still low confidence: prompt user for more images or a short video (≤60s) with guidance.

3. Object Detection Fallbacks

- If no relevant objects found (bottles/labels), crop/focus on likely water areas using segmentation/heuristics → run in-house impurity models.

4. Filters & Enhancements (internal)

- Generate filtered variants (e.g., denoise, contrast, deglare) and compare outcomes; pick the most reliable result while always including original.

5. Potability Score & Interpretation

- Compute Potability 0–100% with factors: visual (turbidity/color), temporal stability (video), context (location/brand/external data), and optional user inputs.
- Bands (initial targets; may evolve with research):
  - 100%: Drinkable — must be corroborated by external sources tied to location and high model confidence.
  - 99%–51%: Very clean, not drinkable.
  - 50%: Clean; usable for daily non-drinking tasks (e.g., dishwashing); not drinkable.
  - 49%–26%: Less clean; not recommended for daily use; use carefully.
  - <25%: Unclean; do not use.
- Always display attribution and caveats (informational only; no safety guarantees).

6. Debug Timeline (when enabled)

- Show per-step confidences, selected frames, filters comparison, OCR snippets, external data sources used, and any fallbacks taken.

Optional user input prompt (shown pre- or mid-analysis):

- "Tell us about the water" — free text and quick selectors for smell/scent, color, feel, temperature.
- This input refines turbidity/temperature inference and final explanation.

## Technical Context

**Language/Version**: Python 3.x only (unified)
**Primary Dependencies**: FastAPI (API), Gradio (Web UI), SQLAlchemy/SQLite, Pydantic v2, httpx.
**ML Libraries**:

- Scene classification: External pretrained Places365 CNNs (ResNet family)
  - ResNet152 (Caffe) deploy prototxt: https://github.com/CSAILVision/places365/blob/master/deploy_resnet152_places365.prototxt
  - ResNet152 weights (Caffe): http://places2.csail.mit.edu/models_places365/resnet152_places365.caffemodel
  - Alternative PyTorch checkpoints: ResNet18/ResNet50/DenseNet161 `.pth.tar` from Places365 models page
- Object detection (baseline): RF-DETR (https://github.com/roboflow/rf-detr, Apache-2.0)
- Object detection (compatibility path): Ultralytics YOLO11 (often referred to as YOLOv11) via `ultralytics` package (AGPL-3.0 / Enterprise)
  - Docs: https://docs.ultralytics.com/models/yolo11/
  - Repo: https://github.com/ultralytics/ultralytics (AGPL-3.0)
- Object detection (in-house training option): YOLOv12 (https://github.com/sunsmarterjie/yolov12, AGPL-3.0)
  **Model Management**:
- Auto-download Places365 model assets from the above URLs during ML bootstrap (opt-out via env) with checksum verification and caching
- Prefer native PyTorch for inference when available; convert/bridge Caffe models as needed (export to ONNX or use a Caffe runtime bridge if required)
  **Storage**: SQLite (for ease of local run); consider local-first history with optional sync later
  **Testing**: Pytest
  **Target Platform**: Web browser (desktop + mobile web)
  **Project Type**: web (Python-only API + Python Gradio UI)
  **Performance Goals**: Usable in a mobile environment with choppy internet. Target image inference <1s on desktop CPU/GPU; 60s video sampled adaptively in a few seconds end-to-end on a development workstation. Exact throughput depends on chosen detector size (RF-DETR N/S/M) or YOLOv12 variant.
  **Constraints**: Video ≤60s; offline-capable analysis fallbacks; privacy-first consents
  Note: Online-only processing for this phase (backend required). Adaptive frame sampling, filter comparison, OCR, and brand detection are automatic. Media limits: ≤25 images and ≤5 videos (≤60s each). Debug levels: none|minimal|detailed. Theming follows OS with Material You-style accents.
  Runtime unification: Launching the Gradio UI (WebInterface) will autostart the FastAPI backend (uvicorn) on localhost; no separate Node/NPM frontend.
  **Scale/Scope**: private testing and not public use, Proof of Concept

Additional UI theme details from user: Glassmorphism theme with Dark/Light modes and configurable accent color.

Optional user prompt inputs

- Fields: smell/scent (free text + preset chips), color (chips + free text), feel (chips), temperature (chips + numeric optional), notes (free text)
- Usage: Incorporated as priors for turbidity/temperature estimation and highlighted in the explanation; stored only if history is enabled.

Regional external water data sources (governmental)

- US: Water Quality Portal (WQP) REST (USGS/EPA) — https://www.waterqualitydata.us/webservices_documentation/; USGS Water Services — https://waterservices.usgs.gov/
- EU: WISE/EEA datasets (river basins and water bodies) — https://www.eea.europa.eu/en/datahub and https://water.europa.eu/
- Indonesia (ID): Open data portal for environmental/water datasets — https://data.go.id/ (curated datasets vary by province/agency)
  Selection rules
- If country detected = US → WQP/USGS; EU → EEA/WISE; ID → data.go.id datasets. If none, skip external context.
- Associate with nearest water body when possible; if only region-level data is available, label the association as regional and warn that "not all places in the region meet the same quality."
- Cache responses with TTL (default: 24h) and apply a search radius default (e.g., 5–25 km depending on dataset granularity). Expose env overrides.

Media constraints and processing

- Video: max duration per video 60s; max 5 videos per session. Downsample/limit analysis to 30 FPS for speed; use adaptive frame sampling on top of this cap.
- Photos: max 25 images per session.
- Filters: apply both to photos and video frames; compare variant outcomes to improve confidence.

Confidence representation

- Internally 0.0–1.0; present to users as 0–100% where 1.0 maps to 100%.
- Show per-step confidences in Debug Mode; show an overall confidence with the result summary.

## Constitution Check

GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.

- Principles are currently placeholders and need ratification content in constitution.md. No violations detected yet, but Test-First and Observability should be planned explicitly (contract tests, structured logs).

## Project Structure

### Source Code (repository root)

```
# Unified Python application (API + Web UI)


WebInterface/
└── app.py              # Gradio UI (Python-only frontend)
└ backend/


In-HouseModel Training/
├── classification/     # YOLOv12 (or current Ultralytics) classification training scaffolding
├── Detection/          # YOLOv12 object detection training scaffolding
```

**Structure Decision**: Unified Python web app (FastAPI + Gradio) with SQLite; NPM frontend removed. WebInterface is the primary entrypoint and is responsible for autostarting the backend.

### External Models and Training (decisions)

- Scene classification uses external pretrained Places365 ResNet models (not in-house). Primary source links are embedded above; the system will auto-download the specified assets. If using the Caffe ResNet152 model, plan to either: (a) use a conversion pipeline (Caffe → ONNX → PyTorch) or (b) run via a Caffe runtime shim. Prefer PyTorch ResNet50 from Places365 when available to avoid conversion friction.
- Object detection will use RF-DETR as the default detector for inference and fine-tuning (Apache-2.0). Provide training hooks via the `rfdetr` package.
- Detector compatibility: We support multiple detector backends behind a single adapter interface. Choices:
  - RF-DETR (default, Apache-2.0)
  - Ultralytics YOLO11 (YOLOv11) for compatibility (AGPL-3.0 / Enterprise)
  - YOLOv12 (AGPL-3.0) for in-house training and experimentation
    Selection via env var `HYDROSCAN_DETECTOR_BACKEND=rfdetr|yolo11|yolo12` (default: `rfdetr`).
- YOLOv12 will be supported as an alternative training pathway for object detection with clear AGPL-3.0 license implications; use only if license is acceptable for the deployment scenario. For commercial deployments of Ultralytics software/models (including YOLO11), obtain an Enterprise license per their terms or avoid including the `ultralytics` dependency.

## Detector Compatibility & Backends

We provide a unified detector adapter with a stable output schema regardless of the underlying backend to make swapping detectors zero-touch for the rest of the app.

- Env selection

  - HYDROSCAN_DETECTOR_BACKEND: rfdetr | yolo11 | yolo12 (default: rfdetr)
  - HYDROSCAN_DETECTOR_MODEL: filesystem path or model name (per-backend semantics)
    - rfdetr: path to .pth/.pt checkpoint or hub name
    - yolo11: path to Ultralytics .pt or model alias like "yolo11n.pt"
    - yolo12: path to in-house trained weights (.pt) or exported ONNX
  - HYDROSCAN_DETECTOR_DEVICE: cpu | cuda | cuda:0 (optional)
  - HYDROSCAN_ML_SKIP_DOWNLOAD: true to disable auto-downloading external assets

- Adapter contract (Python)

  - predict(images: list[np.ndarray], conf: float = 0.25, iou: float = 0.45) -> list[list[Detection]]
  - Detection dataclass: { bbox_xyxy: tuple[float, float, float, float], cls: str, score: float, track_id: Optional[int] }
  - Coordinates are absolute pixel xyxy in the input image space; callers can request normalization if needed.

- Output schema (API/JSON)

  - detections: [ { x1, y1, x2, y2, class, score } ... ] per image/frame
  - classes: optional dictionary { class_id/name -> metadata }
  - model: { name, version, backend, device }

- Licensing notes

  - RF-DETR: Apache-2.0. Safe for commercial use.
  - Ultralytics YOLO11 (YOLOv11): AGPL-3.0 or Enterprise. Using the `ultralytics` package or distributing its models may trigger AGPL obligations; obtain an Enterprise license for closed-source commercial deployments.
  - YOLOv12: AGPL-3.0. Treat similarly to YOLO11 regarding redistribution/commercialization.
  - Optional ONNX export can decouple runtime, but training/code that produced the weights may still impose license obligations. Consult counsel for your use case.

- Model formats
  - Native PyTorch: .pt / .pth
  - ONNX: .onnx (preferred for interop where possible)
  - We will provide light wrappers to load/export where supported by each backend.

### Analysis Workflow to Implementation Mapping

- Scene detection → Places365-based classifier → user confirmation override when uncertain.
- Outdoor path → location consent → external data fetch (with caching) → combine with media analysis; fallback to in-house only if external data absent.
- Indoor path → detector adapter (RF-DETR default; YOLO11/YOLOv12 optional) → OCR (brand) → brand lookup → fallback to media-only.
- Water-region focus → crop/segment pipeline when objects not detected or branding unknown.
- Filters/variants → run analysis on original + variants; compare and choose most reliable.
- Potability score → combine visual cues, temporal stability (video), external data, and optional user inputs; attribute all sources.
- Debug timeline → structured logging + UI panel to render per-step artifacts.

Brand label workflow

- Detection → OCR → Brand search: attempt public knowledge sources first; then web search; if necessary, scraping is allowed within robots.txt and rate limits; always show attribution and avoid storing scraped PII.
- Env toggles: HYDROSCAN_BRAND_WEBSEARCH=on/off, HYDROSCAN_BRAND_SCRAPING=on/off, HYDROSCAN_BRAND_RATE_LIMIT (req/min), HYDROSCAN_BRAND_PROVIDERS (ordered list).

## Phase 0: Outline & Research

1. Extract unknowns from Technical Context above (RESOLVED):
   - Stack: Unified Python (FastAPI + Gradio)
   - Storage: SQLite local-first
   - Scene model: External Places365 (ResNet) with auto-download and conversion plan
   - Detector: RF-DETR (baseline, Apache-2.0); YOLOv12 optional (AGPL-3.0)
   - Accessibility: Aim WCAG AA; keyboard navigation and non-color cues
   - Scoring: Potability 0–100% with bands; thresholds aligned with spec
   - Privacy: Online backend for this phase; no media persistence unless history enabled
   - Performance: Targets noted above
2. Research captured in `research.md` (complete) with rationale for model choices and licensing notes (Apache-2.0 vs AGPL-3.0 trade-offs).

Output: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts

1. Extract entities → `data-model.md` (MediaItem, SceneClassification, PotabilityScore, etc.) with fields and relationships
2. Generate API contracts (REST) from FRs:
   - /analyze (media upload/URL), /status, /results/{id}, /brands/lookup, /context/location
   - Output OpenAPI spec in `/contracts/`
   - Gradio UI will call these endpoints; local in-process mode is not required for production
   - Autostart: WebInterface will spawn uvicorn for the API if not running
3. Generate contract tests (failing) for each endpoint
4. Extract test scenarios to `quickstart.md` (happy paths + edge cases)
5. Update agent file via `.specify/scripts/powershell/update-agent-context.ps1 -AgentType copilot` with new tech

Output: data-model.md, /contracts/\*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach

- Use `.specify/templates/tasks-template.md` as base; generate tasks from contracts, data model, quickstart
- TDD order; models → services → UI (Gradio) → integration; mark independent tasks as [P]
- Include ML tasks: implement an auto-downloader that fetches Places365 assets from the exact URLs listed; add checksum verification and caching; implement detector adapters for RF-DETR, YOLO11, and YOLOv12; selection via env; add basic integration tests that exercise the adapter interface (skip if a backend package isn't installed); plan YOLOv12 training entrypoints where licensing permits and document YOLO11 Enterprise option.

MVP definition and legal posture

- MVP: A working prototype that runs locally with: scene classification; the branching workflow; detector adapter (RF-DETR by default); OCR+brand lookup with optional web search/scraping; external data integration for US/EU/ID; potability scoring and debug timeline; video/photo constraints enforced; confidence 0–1 reported; history optional.
- Private prototype only: not for public release. Analysis is informational; no health/safety guarantees. Licensing: RF-DETR (Apache-2.0) default; Ultralytics and YOLOv12 (AGPL-3.0) optional and not required for MVP.
- Estimated: 30–35 tasks reflecting unified UI and concrete ML integration (no stubs).

## Clarifications Needed (to finalize design)

- External data sources for outdoor water quality:
  - Which primary APIs to use? Examples: EPA WQP/WaterNow (US), EU Water Framework Directive endpoints, local government datasets, WHO/UNICEF JMP for general context.
  - Acceptable default: region-scoped open datasets; define fallback search rules and max latency per call.
  - Required: TTL for cache (e.g., 24h?) and search radius defaults (e.g., 5–25 km, environment-specific).
- Brand/label knowledge sources:
  - Which datasets or web endpoints are acceptable? Rate limits and scraping policy constraints.
  - Multilingual OCR: which languages must be supported in MVP? What is the fallback when OCR confidence is low?
- Scoring specifics:
  - Exact formula weights for Potability components (visual cues vs external data vs user inputs) and capping behavior for 100%.
  - Are the provided percentage bands fixed for MVP, or allowed to adjust after initial validation?
- User inputs:
  - Required presets for smell/color/feel/temperature; any prohibited terms? Units for temperature (°C/°F)?
  - Privacy: confirm that storing optional descriptions is controlled solely by the history toggle.
- Video handling:
  - Target FPS for adaptive sampling and max resolution to process; timeouts and memory constraints.
- Debug Mode:
  - Which artifacts are safe to store/show (privacy) and how long to retain in history?
- Accessibility & UI:
  - Target baseline (WCAG 2.1 AA?) and any additional constraints.
- Legal text:
  - Provide the disclaimer language for “informational only, not a safety/health guarantee.”

## Complexity Tracking

| Violation                    | Why Needed                                                      | Simpler Alternative Rejected Because                  |
| ---------------------------- | --------------------------------------------------------------- | ----------------------------------------------------- |
| Separate frontend/backend    | (Deprecated) moving to unified Python UI for simplicity         | Keeping separate stacks adds overhead and duplication |
| In-process UI using services | Allows local, offline-friendly flows and simpler dev experience | Pure API-only requires another UI stack               |
