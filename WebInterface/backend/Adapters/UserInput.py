from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Optional heavyweight dependency
    from llama_cpp import Llama  # type: ignore

    _HAS_LLAMA_CPP = True
except Exception:  # pragma: no cover - optional dependency
    Llama = None  # type: ignore
    _HAS_LLAMA_CPP = False


@dataclass(slots=True)
class UserInputAssessment:
    available: bool
    conclusion: str
    score: float
    confidence: float
    rationale: str | None = None
    model_name: str | None = None
    reason: str | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "conclusion": self.conclusion,
            "score": self.score,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "model_name": self.model_name,
            "reason": self.reason,
        }


class UserInputAdapter:
    """Optional adapter that uses a local LLaMA 3 model to interpret user descriptions."""

    def __init__(
        self,
        models_dir: Optional[str] = None,
        model_filename: Optional[str] = None,
    ) -> None:
        self.models_dir = Path(models_dir or "")
        if not self.models_dir:
            self.models_dir = (
                Path(__file__).resolve().parents[2] / "Models" / "UserInput"
            )
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model_filename = model_filename or os.environ.get(
            "HYDROSCAN_USER_MODEL_NAME"
        )
        self.model_path = self._resolve_model_path()
        self.model_name = os.environ.get("LLAMA3_MODEL_NAME", "LLaMA-3")
        self._status: str | None = None
        self._llama: Llama | None = None
        self.available: bool = False

        if not self.model_path:
            self._status = "No LLaMA model file found"
            return

        if not _HAS_LLAMA_CPP:
            self._status = "llama-cpp-python is not installed"
            return

        try:
            n_ctx = int(os.environ.get("HYDROSCAN_USER_MODEL_CTX", "4096"))
            n_threads = int(os.environ.get("HYDROSCAN_USER_MODEL_THREADS", "4"))
            n_gpu_layers = int(os.environ.get("HYDROSCAN_USER_MODEL_GPU_LAYERS", "0"))
            self._llama = Llama(  # type: ignore[call-arg]
                model_path=str(self.model_path),
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                embedding=False,
            )
            self.available = True
        except Exception as exc:  # pragma: no cover - hardware/runtime specific
            self._status = f"Failed to load LLaMA model: {exc}"
            self._llama = None
            self.available = False

    # ------------------------------------------------------------------
    def _resolve_model_path(self) -> Optional[Path]:
        if self.model_filename:
            candidate = self.models_dir / self.model_filename
            if candidate.exists():
                return candidate

        preferred_suffixes = [".gguf", ".ggml", ".bin"]
        for suffix in preferred_suffixes:
            matches = sorted(self.models_dir.glob(f"*{suffix}"))
            if matches:
                return matches[0]
        return None

    # ------------------------------------------------------------------
    def analyze(self, description: str, context: Dict[str, Any]) -> UserInputAssessment:
        if not description:
            return UserInputAssessment(
                available=False,
                conclusion="",
                score=45.0,
                confidence=40.0,
                rationale=None,
                reason="No user description provided",
            )

        if not self.available or not self._llama:
            return UserInputAssessment(
                available=False,
                conclusion="",
                score=45.0,
                confidence=40.0,
                rationale=None,
                reason=self._status or "User input model not available",
            )

        prompt = self._build_prompt(description, context)
        max_tokens = int(os.environ.get("HYDROSCAN_USER_MODEL_MAX_TOKENS", "256"))
        temperature = float(os.environ.get("HYDROSCAN_USER_MODEL_TEMPERATURE", "0.2"))

        try:
            result = self._llama.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</analysis>"],
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            return UserInputAssessment(
                available=False,
                conclusion="",
                score=45.0,
                confidence=40.0,
                rationale=None,
                reason=f"LLaMA inference failed: {exc}",
            )

        text = ""
        try:
            choices = result.get("choices") or []
            if choices:
                text = choices[0].get("text", "")
        except Exception:
            text = ""

        assessment = self._parse_response(text)
        assessment.model_name = self.model_name or self.model_path.name
        return assessment

    # ------------------------------------------------------------------
    def _build_prompt(self, description: str, context: Dict[str, Any]) -> str:
        detections = context.get("detections", [])
        top_detection = context.get("top_detection") or {}
        visual_metrics = context.get("visual_metrics") or {}
        external = context.get("external") or {}
        scene = context.get("scene") or "unknown"
        base_scores = context.get("base_scores") or {}

        detection_summary = (
            f"{len(detections)} detections; "
            f"top class: {top_detection.get('class_name', 'n/a')} "
            f"(score={top_detection.get('score', 0.0):.2f})"
            if detections
            else "No water detections identified"
        )
        external_summary = (
            f"station {external.get('station_id')} quality {external.get('overall_quality')}"
            if external
            else "No external station data"
        )
        base_score_summary = (
            f"potability={base_scores.get('potability_score', 'n/a')} "
            f"confidence={base_scores.get('confidence_score', 'n/a')}"
            if base_scores
            else "Scores not yet computed"
        )

        prompt = (
            "### System\n"
            "You are HydroScan's senior water quality analyst. Assess user observations in light of objective analysis.\n"
            "Return a strict JSON object with the following keys:\n"
            '  "conclusion": short plain-language summary (max 2 sentences) relating the water safety.\n'
            '  "score": integer 0-100 representing the user-text contribution to potability (higher is safer).\n'
            '  "confidence": integer 0-100 reflecting how confident you are in the user-text signal.\n'
            '  "rationale": optional short explanation (max 3 sentences).\n'
            "Enclose nothing outside the JSON and do not include markdown.\n\n"
            "### Watched Context\n"
            f"- Scene majority: {scene}\n"
            f"- Detection summary: {detection_summary}\n"
            f"- Visual metrics: {json.dumps(visual_metrics, ensure_ascii=False)}\n"
            f"- External data: {external_summary}\n"
            f"- Baseline scores: {base_score_summary}\n\n"
            "### User Description\n"
            f"{description.strip()}\n\n"
            "### Response\n"
            '{"conclusion": '
        )
        return prompt

    # ------------------------------------------------------------------
    def _parse_response(self, text: str) -> UserInputAssessment:
        cleaned = text.strip()
        if not cleaned:
            return UserInputAssessment(
                available=False,
                conclusion="",
                score=45.0,
                confidence=40.0,
                rationale=None,
                reason="Empty response from LLaMA",
            )

        # Ensure valid JSON. The prompt starts the object with {"conclusion":
        if not cleaned.endswith("}"):
            cleaned = cleaned.split("}", 1)[0] + "}"

        try:
            payload = json.loads(cleaned)
        except Exception:
            return UserInputAssessment(
                available=False,
                conclusion="",
                score=45.0,
                confidence=40.0,
                rationale=cleaned[:280],
                reason="Unable to parse LLaMA JSON response",
            )

        conclusion = str(payload.get("conclusion", "")).strip()
        score = float(payload.get("score", 45.0))
        confidence = float(payload.get("confidence", 40.0))
        rationale = payload.get("rationale")
        if isinstance(rationale, str):
            rationale = rationale.strip()
        else:
            rationale = None

        score = max(0.0, min(100.0, score))
        confidence = max(0.0, min(100.0, confidence))

        return UserInputAssessment(
            available=True,
            conclusion=conclusion,
            score=score,
            confidence=confidence,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    @property
    def status(self) -> str:
        if self.available:
            return "ready"
        return self._status or "uninitialized"
