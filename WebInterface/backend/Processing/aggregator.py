from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _iou(box_a: List[float], box_b: List[float]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    union = area_a + area_b - inter_area + 1e-6
    return float(inter_area / union)


def _merge_detections(
    aggregated: List[Dict[str, Any]],
    new_detections: List[Dict[str, Any]],
    frame_index: int | None,
    variant: str | None,
    iou_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    def _score(det: Dict[str, Any]) -> float:
        candidate = det.get("score")
        if candidate is None:
            candidate = det.get("confidence")
        if candidate is None:
            candidate = det.get("probability")
        try:
            return float(candidate)
        except Exception:
            return 0.0

    for det in new_detections:
        bbox = det.get("bbox") or det.get("box")
        if not bbox:
            continue

        # Convert OBB (8-element) to regular bbox (4-element) if needed
        if len(bbox) == 8:
            # OBB format: [x1, y1, x2, y2, x3, y3, x4, y4] -> convert to [x_min, y_min, x_max, y_max]
            x_coords = [bbox[0], bbox[2], bbox[4], bbox[6]]
            y_coords = [bbox[1], bbox[3], bbox[5], bbox[7]]
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            det["bbox"] = bbox  # Update the detection with converted bbox

        matched = False
        for agg in aggregated:
            if agg.get("class_id") != det.get("class_id"):
                continue
            agg_bbox = agg.get("bbox", [])
            # Convert aggregated bbox if needed
            if len(agg_bbox) == 8:
                x_coords = [agg_bbox[0], agg_bbox[2], agg_bbox[4], agg_bbox[6]]
                y_coords = [agg_bbox[1], agg_bbox[3], agg_bbox[5], agg_bbox[7]]
                agg_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                agg["bbox"] = agg_bbox

            if _iou(agg_bbox, bbox) >= iou_threshold:
                det_score = _score(det)
                agg["score"] = max(float(agg.get("score", 0.0)), det_score)
                occurrences = agg.setdefault("occurrences", [])
                occurrences.append(
                    {
                        "frame_index": frame_index,
                        "variant": variant,
                        "score": det_score,
                        "source": det.get("source"),
                    }
                )
                matched = True
                break

        if not matched:
            det_score = _score(det)
            aggregated.append(
                {
                    "bbox": bbox,
                    "class_id": det.get("class_id"),
                    "class_name": det.get("class_name") or det.get("label"),
                    "score": det_score,
                    "source": det.get("source"),
                    "occurrences": [
                        {
                            "frame_index": frame_index,
                            "variant": variant,
                            "score": det_score,
                            "source": det.get("source"),
                        }
                    ],
                }
            )

    return aggregated


def aggregate(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-frame/variant outputs into a concise summary structure."""

    if not frames:
        return {
            "frames": [],
            "detections": [],
            "class_counts": {},
            "metrics_avg": {},
            "metrics_best": {},
        }

    aggregated_detections: List[Dict[str, Any]] = []
    class_counts: defaultdict[str, int] = defaultdict(int)
    metrics_sum: defaultdict[str, float] = defaultdict(float)
    metrics_best: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    metric_frames = 0

    for frame in frames:
        variant = frame.get("variant")
        frame_index = frame.get("frame_index")

        detections = frame.get("detections") or []
        aggregated_detections = _merge_detections(
            aggregated_detections, detections, frame_index, variant
        )
        for det in detections:
            label = det.get("class_name") or det.get("class_id") or "unknown"
            class_counts[str(label)] += 1

        metrics = frame.get("metrics") or {}
        if metrics:
            metric_frames += 1
        for name, value in metrics.items():
            value_f = float(value)
            metrics_sum[name] += value_f
            best = metrics_best.get(name)
            if best is None or value_f > best[0]:
                metrics_best[name] = (
                    value_f,
                    {
                        "variant": variant,
                        "frame_index": frame_index,
                    },
                )

    metrics_avg = {
        name: metrics_sum[name] / metric_frames if metric_frames else 0.0
        for name in metrics_sum
    }

    metrics_best_summary = {
        name: {"value": val[0], **val[1]} for name, val in metrics_best.items()
    }

    top_detection = max(
        aggregated_detections, key=lambda d: d.get("score", 0.0), default=None
    )

    return {
        "frames": frames,
        "detections": aggregated_detections,
        "top_detection": top_detection,
        "class_counts": dict(class_counts),
        "metrics_avg": metrics_avg,
        "metrics_best": metrics_best_summary,
    }
