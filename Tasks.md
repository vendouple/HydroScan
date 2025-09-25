# Tasks Tracker — HydroScan

This file tracks implementation progress aligned with `plan.md`.

## Bootstrap (this repo)

- [x] Update plan.md with clarifications and final layout
- [x] Scaffold project structure under `WebInterface/`
- [x] Minimal FastAPI app with UI and health
- [x] Model fetcher utility (robust retries + manual fallback)
- [x] Ingestion (Indonesia) scaffolding
- [x] Processing pipeline stubs (filters, metrics, aggregator)
- [x] Scoring engine stub
- [x] Adapters stubs (Places365, RF-DETR, InModel)
- [x] Analyze flow wiring and persistence to `history/`

## Next

- Implement adapters and pipeline incrementally
- Add tests and example media
- Polish UI (timeline, debug console, results view)
