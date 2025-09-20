# Quickstart

1. Create venv and install deps from `requirements.txt`.
2. Run the UI: `python WebInterface/app.py`.
3. Upload one or more images and/or a short video (≤60s). Optionally add notes.
4. Read the result summary and timeline.

## Scenarios

- Image of river (outdoor): expect scene Outdoor/Natural.
- Indoor bottled water label: detector + OCR should pick text; brand lookup via Wikipedia API.
- No detections: water-region heuristics not yet implemented; fall back to scene + filters.
