# Tasks Tracker — HydroScan

This file tracks implementation progress aligned with `plan.md`.

## Bootstrap (this repo)

- [x] Update plan.md with clarifications and final layout
- [x] Scaffold project structure under `WebInterface/`
- [x] Minimal FastAPI app with UI and health
- [ ] Model fetcher utility (robust retries + manual fallback)
- [ ] Ingestion (Indonesia) scaffolding
- [ ] Processing pipeline stubs (filters, metrics, aggregator)
- [ ] Scoring engine stub
- [ ] Adapters stubs (Places365, RF-DETR, InModel)
- [ ] Analyze flow wiring and persistence to `history/`

## Next

- Implement adapters and pipeline incrementally
- Add tests and example media
- Polish UI (timeline, debug console, results view)
