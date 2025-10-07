from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests import Session

try:  # Optional heavyweight dependency for local fallback
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
    """Adapter that prefers Gemini 2.5 Pro (thinking) with optional local LLaMA fallback."""

    GEMINI_ENDPOINT = (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )

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

        self._status: str | None = None
        self.available: bool = False
        self.provider: Optional[str] = None
        self.model_filename = model_filename or os.environ.get(
            "HYDROSCAN_USER_MODEL_NAME"
        )
        self.model_name: Optional[str] = None
        self._llama: Optional[Any] = None
        self.model_path: Optional[Path] = None

        # Gemini configuration
        self.api_key = self._load_api_key()
        self.session: Optional[Session] = None
        self.temperature = float(os.environ.get("HYDROSCAN_GEMINI_TEMPERATURE", "0.1"))
        self.timeout = float(os.environ.get("HYDROSCAN_GEMINI_TIMEOUT", "30"))
        self.thinking_budget = self._parse_int_env(
            "HYDROSCAN_GEMINI_THINKING_BUDGET", default=-1
        )
        self.include_thoughts = self._parse_bool_env(
            "HYDROSCAN_GEMINI_INCLUDE_THOUGHTS", default=False
        )

        if self.api_key:
            self.session = requests.Session()
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
            self.model_name = os.environ.get("HYDROSCAN_GEMINI_MODEL", "gemini-2.5-pro")
            self.available = True
            self.provider = "gemini"
            self._status = "ready"
            return

        # Optional local LLaMA fallback if Gemini is unavailable
        self.model_path = self._resolve_model_path()
        if not self.model_path:
            self._status = "No Gemini key or LLaMA model found"
            return

        if not _HAS_LLAMA_CPP:
            self._status = "Gemini key missing and llama-cpp-python not installed"
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
            self.model_name = os.environ.get("LLAMA3_MODEL_NAME", "LLaMA-3")
            self.available = True
            self.provider = "llama"
            self._status = "ready"
        except Exception as exc:  # pragma: no cover - hardware/runtime specific
            self._status = f"Failed to load fallback LLaMA model: {exc}"
            self._llama = None
            self.available = False

    # ------------------------------------------------------------------
    def _load_api_key(self) -> Optional[str]:
        env_candidates = [
            os.environ.get("HYDROSCAN_GEMINI_API_KEY"),
            os.environ.get("GEMINI_API_KEY"),
            os.environ.get("GOOGLE_GEMINI_API_KEY"),
            os.environ.get("GeminiApiKey"),
        ]
        for candidate in env_candidates:
            if candidate:
                return candidate.strip()

        env_path = self.models_dir / "gemini_key.env"
        if not env_path.exists():
            return None

        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip().lower() == "geminiapikey" and value.strip():
                    return value.strip()
        except Exception:
            return None
        return None

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

    @staticmethod
    def _parse_bool_env(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _parse_int_env(name: str, default: Optional[int] = None) -> Optional[int]:
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    # ------------------------------------------------------------------
    def analyze(self, description: str, context: Dict[str, Any]) -> UserInputAssessment:
        if not description:
            return UserInputAssessment(
                available=False,
                conclusion="",
                score=0.0,
                confidence=0.0,
                rationale=None,
                reason="No user description provided",
            )

        if not self.available:
            return UserInputAssessment(
                available=False,
                conclusion="",
                score=0.0,
                confidence=0.0,
                rationale=None,
                reason=self._status or "User input model not available",
            )

        if self.provider == "gemini" and self.session:
            return self._analyze_with_gemini(description, context)

        if self.provider == "llama" and self._llama:
            return self._analyze_with_llama(description, context)

        return UserInputAssessment(
            available=False,
            conclusion="",
            score=0.0,
            confidence=0.0,
            rationale=None,
            reason=self._status or "User input model unavailable",
        )

    # ------------------------------------------------------------------
    def _analyze_with_gemini(
        self, description: str, context: Dict[str, Any]
    ) -> UserInputAssessment:
        assert self.session is not None  # for type checkers

        messages = self._build_messages(description, context)
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }

        thinking_config: Dict[str, Any] = {}
        if self.thinking_budget is not None:
            thinking_config["thinking_budget"] = self.thinking_budget
        if self.include_thoughts:
            thinking_config["include_thoughts"] = True
        if thinking_config:
            payload["extra_body"] = {"google": {"thinking_config": thinking_config}}

        try:
            response = self.session.post(
                self.GEMINI_ENDPOINT,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:  # pragma: no cover - network specific
            return UserInputAssessment(
                available=False,
                conclusion="",
                score=0.0,
                confidence=0.0,
                rationale=None,
                model_name=self.model_name,
                reason=f"Gemini request failed: {exc}",
            )

        if response.status_code >= 400:
            detail = response.text.strip()
            return UserInputAssessment(
                available=False,
                conclusion="",
                score=0.0,
                confidence=0.0,
                rationale=None,
                model_name=self.model_name,
                reason=f"Gemini HTTP {response.status_code}: {detail[:280]}",
            )

        data: Dict[str, Any]
        try:
            data = response.json()
        except ValueError:
            return UserInputAssessment(
                available=False,
                conclusion="",
                score=0.0,
                confidence=0.0,
                rationale=response.text[:280],
                model_name=self.model_name,
                reason="Gemini response was not valid JSON",
            )

        text = self._extract_message_text(data)
        assessment = self._parse_response(text)
        assessment.model_name = self.model_name
        return assessment

    # ------------------------------------------------------------------
    def _analyze_with_llama(
        self, description: str, context: Dict[str, Any]
    ) -> UserInputAssessment:
        assert self._llama is not None  # for type checkers

        prompt = self._compose_prompt(description, context)
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
                score=0.0,
                confidence=0.0,
                rationale=None,
                model_name=self.model_name,
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
        assessment.model_name = self.model_name or (
            self.model_path.name if self.model_path else None
        )
        return assessment

    # ------------------------------------------------------------------
    def _extract_message_text(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            fragments = []
            for part in content:
                if isinstance(part, dict):
                    fragments.append(str(part.get("text", "")))
                else:
                    fragments.append(str(part))
            return "".join(fragments)
        return str(content or "")

    # ------------------------------------------------------------------
    def _build_messages(
        self, description: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        system_prompt = (
            "You are HydroScan's senior water quality analyst. "
            "Respond with a strict JSON object containing the keys "
            "conclusion (string, max 2 sentences), score (integer 0-100), "
            "confidence (integer 0-100), and rationale (optional string up to 3 sentences). "
            "Do not include markdown or additional text."
        )

        user_prompt = self._compose_prompt(description, context)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # ------------------------------------------------------------------
    def _compose_prompt(self, description: str, context: Dict[str, Any]) -> str:
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
            "Context for assessment:\n"
            f"- Scene majority: {scene}\n"
            f"- Detection summary: {detection_summary}\n"
            f"- Visual metrics: {json.dumps(visual_metrics, ensure_ascii=False)}\n"
            f"- External data: {external_summary}\n"
            f"- Baseline scores: {base_score_summary}\n\n"
            "User description:\n"
            f"{description.strip()}\n\n"
            "Return only the JSON object now."
        )
        return prompt

    # ------------------------------------------------------------------
    def _parse_response(self, text: str) -> UserInputAssessment:
        cleaned = text.strip()
        if not cleaned:
            return UserInputAssessment(
                available=False,
                conclusion="",
                score=0.0,
                confidence=0.0,
                rationale=None,
                reason="Empty response from user input model",
            )

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        # Ensure valid JSON by trimming to outermost braces
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1:
            cleaned = cleaned[first_brace : last_brace + 1]

        try:
            payload = json.loads(cleaned)
        except Exception:
            return UserInputAssessment(
                available=False,
                conclusion="",
                score=0.0,
                confidence=0.0,
                rationale=cleaned[:280],
                reason="Unable to parse JSON response",
            )

        conclusion = str(payload.get("conclusion", "")).strip()
        score = float(payload.get("score", 0.0))
        confidence = float(payload.get("confidence", 0.0))
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
            provider = self.provider or "unknown"
            return f"ready ({provider})"
        return self._status or "uninitialized"
