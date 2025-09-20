# Data Model (Prototype)

- MediaItem: { id, path, type: image|video, created_at }
- Job: { id, media: [paths], user_inputs, debug, status, result }
- SceneClassification: { scene: Outdoor/Natural | Indoor/Unknown, confidence, top5[] }
- Detection: { x1,y1,x2,y2, class, score }
- TimelineStep: { step, ...freeform }
- Result: { id, potability (0-100), confidence (0-100), scene, timeline[] }

SQLite is used via a lightweight table `jobs` storing media paths and JSON results.
