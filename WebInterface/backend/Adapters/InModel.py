from typing import List, Dict, Any, Optional, Tuple
import os
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import cv2


class InModelAdapter:
    """
    In-house YOLOv11 adapter for water quality detection.

    Uses two separate fine-tuned YOLOv11 models from /Models/CustomModel:
    - CLS.pt: Classification model (clean vs dirty water)
    - OBB.pt: Object Bounding Box model (detecting water quality indicators)

    Works together with filter algorithms to reach confidence thresholds through
    adaptive processing and detection enhancement.
    """

    # Water quality specific class names for object detection
    WATER_QUALITY_CLASSES = [
        "Animals near water",
        "Dead Aquatic Life",
        "Green or blue-green scum",
        "Trash in water",
        "Oil on water surface",
        "Construction activities",
        "Industrial discharge",
        "Murky or muddy water",
        "Foam or bubbles",
        "Unusual water color",
        "Erosion or sedimentation",
        "Floating debris",
        "Chemical stains",
        "Excessive vegetation",
        "Fish kill",
        "Mining activities",
        "Agricultural runoff",
        "Sewage or wastewater",
        "Plastic waste",
        "Metal objects",
        "Natural disasters",
        "Sky",
    ]

    # Classification labels for water quality
    CLASSIFICATION_LABELS = {
        0: "Clean",  # Clean water
        1: "Dirty",  # Dirty/polluted water
        2: "NotWater",  # Unknown/uncertain
    }

    # Thresholds / IDs for classification post-processing
    NOT_WATER_CLASS_ID = 2
    NOT_WATER_CONFIDENCE_THRESHOLD = (
        0.70  # 70% confidence => treat as "no water detected"
    )

    def __init__(self, models_dir: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize InModel adapter with two separate YOLOv11 models.

        Args:
            models_dir: Path to Models directory (will look for Models/CustomModel/)
            device: Device to run models on ('cpu', 'cuda', etc.)
        """
        self.device = device or os.environ.get("YOLO_DEVICE", "cpu")

        # Set up model paths
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            # Default to backend/Models from this file's location
            self.models_dir = Path(__file__).parent.parent / "Models"

        self.custom_model_dir = self.models_dir / "CustomModel"
        self.cls_path = self.custom_model_dir / "CLS.pt"
        self.obb_path = self.custom_model_dir / "OBB.pt"

        # Model instances
        self.classifier = None  # CLS.pt - Water quality classification
        self.obb_detector = None  # OBB.pt - Object detection with bounding boxes
        self.models_loaded = {"classification": False, "obb": False}

        # Load models
        self._load_models()

    def _load_models(self):
        """Load the two in-house YOLOv11 models."""
        try:
            from ultralytics import YOLO

            print(f"[InModel] Looking for models in: {self.custom_model_dir}")
            print(
                f"[InModel] CLS.pt path: {self.cls_path} (exists: {self.cls_path.exists()})"
            )
            print(
                f"[InModel] OBB.pt path: {self.obb_path} (exists: {self.obb_path.exists()})"
            )

            # Load classification model (CLS.pt)
            if self.cls_path.exists():
                try:
                    self.classifier = YOLO(str(self.cls_path))
                    if self.device:
                        self.classifier.to(self.device)
                    self.models_loaded["classification"] = True
                    print(f"[InModel] ✅ Loaded classification model: {self.cls_path}")
                except Exception as e:
                    print(f"[InModel] ❌ Failed to load CLS.pt: {e}")
                    import traceback

                    print(f"[InModel] CLS.pt traceback: {traceback.format_exc()}")
            else:
                print(f"[InModel] ⚠️ Classification model not found: {self.cls_path}")

            # Load OBB detection model (OBB.pt)
            if self.obb_path.exists():
                try:
                    self.obb_detector = YOLO(str(self.obb_path))
                    if self.device:
                        self.obb_detector.to(self.device)
                    self.models_loaded["obb"] = True
                    print(f"[InModel] ✅ Loaded OBB model: {self.obb_path}")
                except Exception as e:
                    print(f"[InModel] ❌ Failed to load OBB.pt: {e}")
                    import traceback

                    print(f"[InModel] OBB.pt traceback: {traceback.format_exc()}")
            else:
                print(f"[InModel] ⚠️ OBB model not found: {self.obb_path}")

        except ImportError as e:
            print(f"[InModel] ❌ Ultralytics YOLO not available: {e}")
            import traceback

            print(f"[InModel] Import traceback: {traceback.format_exc()}")
        except Exception as e:
            print(f"[InModel] ❌ Error loading models: {e}")
            import traceback

            print(f"[InModel] General traceback: {traceback.format_exc()}")

    def classify_water(self, image: Image.Image, conf: float = 0.25) -> Dict[str, Any]:
        """
        Classify water quality using CLS.pt model.

        Args:
            image: PIL Image to classify
            conf: Confidence threshold

        Returns:
            Dict with classification result
        """
        if not self.classifier:
            return {
                "success": False,
                "error": "Classification model not loaded",
                "classification": "unknown",
                "confidence": 0.0,
            }

        try:
            results = self.classifier(image, verbose=False, conf=conf)
            if not results or len(results) == 0:
                return {
                    "success": False,
                    "error": "No classification results",
                    "classification": "unknown",
                    "confidence": 0.0,
                }

            result = results[0]
            probs = getattr(result, "probs", None)

            if probs is None:
                return {
                    "success": False,
                    "error": "No probabilities in result",
                    "classification": "unknown",
                    "confidence": 0.0,
                }

            prob_tensor = getattr(probs, "data", None)
            if prob_tensor is None:
                prob_tensor = probs

            prob_values: List[float] = []
            if hasattr(prob_tensor, "cpu"):
                try:
                    prob_values = prob_tensor.cpu().numpy().tolist()  # type: ignore[attr-defined]
                except Exception:
                    prob_values = [float(x) for x in list(prob_tensor)]
            else:
                prob_values = [float(x) for x in list(prob_tensor)]

            if not prob_values:
                return {
                    "success": False,
                    "error": "Probability vector empty",
                    "classification": "unknown",
                    "confidence": 0.0,
                }

            breakdown = []
            probabilities: Dict[str, float] = {}
            for idx, value in enumerate(prob_values):
                label = self.CLASSIFICATION_LABELS.get(idx, f"class_{idx}")
                score = float(value)
                percent = round(score * 100.0, 2)
                probabilities[label] = score
                breakdown.append(
                    {
                        "class_id": idx,
                        "class_name": label,
                        "score": score,
                        "percent": percent,
                    }
                )

            breakdown.sort(key=lambda item: item["score"], reverse=True)
            top_entry = breakdown[0]
            top_idx = int(top_entry["class_id"])
            confidence = float(top_entry["score"])
            classification = top_entry["class_name"]
            confidence_percent = top_entry["percent"]

            return {
                "success": True,
                "classification": classification,
                "confidence": confidence,
                "confidence_percent": confidence_percent,
                "confidence_label": f"{confidence_percent:.1f}%",
                "score": confidence,
                "class_id": top_idx,
                "probabilities": probabilities,
                "probability_breakdown": breakdown,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Classification failed: {str(e)}",
                "classification": "unknown",
                "confidence": 0.0,
            }

    def detect_objects(
        self, image: Image.Image, conf: float = 0.25, iou: float = 0.45
    ) -> Dict[str, Any]:
        """
        Detect objects using OBB.pt model.

        Args:
            image: PIL Image to analyze
            conf: Confidence threshold
            iou: IoU threshold for NMS

        Returns:
            Dict with detection results
        """
        if not self.obb_detector:
            return {
                "success": False,
                "error": "OBB detection model not loaded",
                "detections": [],
            }

        try:
            results = self.obb_detector(image, verbose=False, conf=conf, iou=iou)
            if not results or len(results) == 0:
                return {
                    "success": False,
                    "error": "No detection results",
                    "detections": [],
                }

            result = results[0]
            detections = []

            # Handle OBB results (Oriented Bounding Boxes)
            if hasattr(result, "obb") and result.obb is not None:
                obb_data = result.obb
                names = getattr(result, "names", {}) or {}

                if hasattr(obb_data, "xyxyxyxy"):
                    boxes = obb_data.xyxyxyxy.cpu().numpy()
                    confidences = obb_data.conf.cpu().numpy()
                    classes = obb_data.cls.cpu().numpy().astype(int)

                    for i, (box, conf_val, cls_id) in enumerate(
                        zip(boxes, confidences, classes)
                    ):
                        class_name = names.get(
                            cls_id,
                            (
                                self.WATER_QUALITY_CLASSES[cls_id]
                                if cls_id < len(self.WATER_QUALITY_CLASSES)
                                else f"class_{cls_id}"
                            ),
                        )

                        score = float(conf_val)
                        detection = {
                            "bbox": box.flatten().tolist(),  # 8 coordinates for OBB
                            "class_id": int(cls_id),
                            "class_name": class_name,
                            "confidence": score,
                            "confidence_percent": round(score * 100.0, 2),
                            "score": score,
                            "source": "inmodel_obb",
                            "type": "obb",
                        }
                        detections.append(detection)

            # Handle regular bounding boxes as fallback
            elif hasattr(result, "boxes") and result.boxes is not None:
                boxes_data = result.boxes
                names = getattr(result, "names", {}) or {}

                if hasattr(boxes_data, "xyxy"):
                    boxes = boxes_data.xyxy.cpu().numpy()
                    confidences = boxes_data.conf.cpu().numpy()
                    classes = boxes_data.cls.cpu().numpy().astype(int)

                    for i, (box, conf_val, cls_id) in enumerate(
                        zip(boxes, confidences, classes)
                    ):
                        class_name = names.get(
                            cls_id,
                            (
                                self.WATER_QUALITY_CLASSES[cls_id]
                                if cls_id < len(self.WATER_QUALITY_CLASSES)
                                else f"class_{cls_id}"
                            ),
                        )

                        score = float(conf_val)
                        detection = {
                            "bbox": box.tolist(),  # 4 coordinates for regular bbox
                            "class_id": int(cls_id),
                            "class_name": class_name,
                            "confidence": score,
                            "confidence_percent": round(score * 100.0, 2),
                            "score": score,
                            "source": "inmodel_detection",
                            "type": "bbox",
                        }
                        detections.append(detection)

            return {"success": True, "detections": detections, "count": len(detections)}

        except Exception as e:
            return {
                "success": False,
                "error": f"Object detection failed: {str(e)}",
                "detections": [],
            }

    def predict_comprehensive(
        self, image: Image.Image, conf: float = 0.25, iou: float = 0.45
    ) -> Dict[str, Any]:
        """
        Comprehensive prediction using both CustomModel models.

        Uses CLS.pt for classification and OBB.pt for object detection.
        Combines results to provide complete water quality analysis.
        """
        results = {
            "success": False,
            "classification": {},
            "classifications": [],
            "detections": [],
            "combined_confidence": 0.0,
            "model_status": self.models_loaded.copy(),
            "probability_breakdown": [],
            "water_found": True,
            "no_water_reason": None,
        }

        # Run classification
        classification_result = self.classify_water(image, conf)
        if classification_result["success"]:
            results["classification"] = classification_result
            results["probability_breakdown"] = classification_result.get(
                "probability_breakdown", []
            )
            classification_entry = {
                "classification": classification_result.get("classification"),
                "class": classification_result.get("classification"),
                "class_name": classification_result.get("classification"),
                "confidence": classification_result.get("confidence", 0.0),
                "score": classification_result.get("confidence", 0.0),
                "class_id": classification_result.get("class_id"),
                "confidence_percent": classification_result.get("confidence_percent"),
                "probability_breakdown": classification_result.get(
                    "probability_breakdown", []
                ),
                "source": "inmodel_classification",
            }
            results["classifications"] = [classification_entry]
            results["success"] = True

            # Determine if we should mark "no water" based on NotWater prediction
            if (
                classification_result.get("class_id") == self.NOT_WATER_CLASS_ID
                and classification_result.get("confidence", 0.0)
                >= self.NOT_WATER_CONFIDENCE_THRESHOLD
            ):
                results["water_found"] = False
                results["no_water_reason"] = (
                    "Classification predicted 'NotWater' with "
                    f"{classification_result.get('confidence_percent', 0.0):.1f}% confidence."
                )

        # Run object detection
        detection_result = self.detect_objects(image, conf, iou)
        if detection_result["success"]:
            results["detections"] = detection_result["detections"]
            results["success"] = True

        # Calculate combined confidence from both models
        confidences = []
        if results["classification"]:
            confidences.append(results["classification"]["confidence"])
        if results["detections"]:
            det_confidences = [det["confidence"] for det in results["detections"]]
            if det_confidences:
                confidences.append(max(det_confidences))

        if confidences:
            results["combined_confidence"] = sum(confidences) / len(confidences)

        return results

    def classify_image_comprehensive(
        self, image: Image.Image
    ) -> Tuple[str, Dict[str, float], Image.Image]:
        """
        Comprehensive water quality analysis using both CustomModel models.

        Returns:
            - status_log: Detailed processing steps as string
            - result_dict: Final classification with confidence
            - annotated_image: Image with bounding boxes drawn
        """
        status = []
        img = image.convert("RGB")

        status.append("🔍 [1] Running CustomModel YOLOv11 water quality detection")
        status.append(
            f"📊 Models loaded: Classification={self.models_loaded['classification']}, OBB={self.models_loaded['obb']}"
        )

        # Get comprehensive predictions using both models
        results = self.predict_comprehensive(image)

        if not results["success"]:
            status.append("⚠️ Detection failed, returning unknown result")
            return "\n".join(status), {"Unknown": 0.0}, img

        # Process classification results
        if results["classification"]:
            cls_result = results["classification"]
            confidence_percent = cls_result.get(
                "confidence_percent", cls_result.get("confidence", 0.0) * 100.0
            )
            status.append(
                "🏷️ [2] Water classification: "
                f"{cls_result['classification']} ({confidence_percent:.1f}%)"
            )

            probability_breakdown = cls_result.get("probability_breakdown", [])
            if probability_breakdown:
                status.append("📈 [2a] Class confidence breakdown:")
                for entry in probability_breakdown:
                    status.append(
                        f"   • {entry['class_name']}: {entry['percent']:.1f}%"
                    )

            if results.get("water_found") is False:
                reason = results.get("no_water_reason")
                status.append(
                    "🚫 [2b] Model is confident this image is not water."
                    + (f" {reason}" if reason else "")
                )
        else:
            status.append("🏷️ [2] Classification model not available")

        # Process detection results
        detection_count = len(results.get("detections", []))
        status.append(f"🎯 [3] Object detections found: {detection_count}")

        # Create annotated image
        arr = np.array(img)
        draw = ImageDraw.Draw(img)

        # Draw detections
        for i, detection in enumerate(results.get("detections", [])):
            if detection.get("type") == "obb" and len(detection["bbox"]) == 8:
                # OBB detection - 8 coordinates
                points = [
                    (detection["bbox"][j], detection["bbox"][j + 1])
                    for j in range(0, 8, 2)
                ]
                draw.polygon(points, outline="red", width=2)
                label = (
                    f"{detection['class_name']} {detection['confidence_percent']:.1f}%"
                )
                draw.text((points[0][0], points[0][1] - 15), label, fill="red")
            else:
                # Regular bounding box - 4 coordinates
                bbox = detection["bbox"]
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                label = (
                    f"{detection['class_name']} {detection['confidence_percent']:.1f}%"
                )
                draw.text((x1, max(y1 - 15, 0)), label, fill="green")

        # Build final result
        result_dict: Dict[str, float] = {}
        if results["classification"]:
            cls_result = results["classification"]
            probability_breakdown = cls_result.get("probability_breakdown", [])

            # Populate per-class scores and percentages for readability
            for entry in probability_breakdown:
                result_dict[entry["class_name"]] = entry["score"]
                result_dict[f"{entry['class_name']}_percent"] = entry["percent"]

            result_dict["TopClassification"] = cls_result["classification"]
            result_dict["TopConfidence"] = cls_result.get("confidence", 0.0)
            result_dict["TopConfidencePercent"] = cls_result.get(
                "confidence_percent", cls_result.get("confidence", 0.0) * 100.0
            )

        result_dict["WaterFound"] = bool(results.get("water_found", True))
        if results.get("no_water_reason"):
            result_dict["NoWaterReason"] = results["no_water_reason"]

        # Add combined confidence
        if results["combined_confidence"] > 0:
            result_dict["Combined_Confidence"] = results["combined_confidence"]
            result_dict["Combined_ConfidencePercent"] = round(
                results["combined_confidence"] * 100.0, 2
            )

        status.append(
            f"✅ [4] Analysis complete. Combined confidence: {results['combined_confidence']:.2f}"
        )

        return "\n".join(status), result_dict, img

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models."""
        return {
            "classification_loaded": self.models_loaded["classification"],
            "obb_loaded": self.models_loaded["obb"],
            "classification_path": (
                str(self.cls_path) if self.cls_path.exists() else None
            ),
            "obb_path": str(self.obb_path) if self.obb_path.exists() else None,
            "device": self.device,
        }

    def predict(
        self, images: List[Image.Image], conf: float = 0.25, iou: float = 0.45
    ) -> List[List[Dict[str, Any]]]:
        """
        Legacy predict method for backward compatibility.
        Uses OBB detector for object detection.
        """
        results = []
        for image in images:
            detection_result = self.detect_objects(image, conf, iou)
            if detection_result["success"]:
                results.append(detection_result["detections"])
            else:
                results.append([])
        return results
