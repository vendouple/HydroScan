"""
Gemini 3 Flash Adapter for Akinator-style water quality inference.
Uses OpenAI-compatible API to communicate with helixmind.online/v1
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# API Configuration
API_KEY = "helix-iZSwTk_jQXJL_AruhsPN3fmkF_dZzrOHbs6etXUGpsg"
BASE_URL = "https://helixmind.online/v1"
MODEL_NAME = "gemini-3-flash-preview"


class ConfidenceLevel(Enum):
    """Confidence levels for predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class AkinatorState:
    """State for an Akinator session."""
    session_id: str
    question_count: int = 0
    max_questions: int = 10
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    detection_context: Dict[str, Any] = field(default_factory=dict)
    user_description: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    cancelled: bool = False


@dataclass
class Question:
    """A question from the Akinator."""
    question_id: str
    question_text: str
    round_number: int
    context: Optional[str] = None


@dataclass
class InferenceResult:
    """Final inference result from the Akinator."""
    predicted_label: str
    confidence: float
    confidence_level: ConfidenceLevel
    reasoning: str
    question_count: int
    detection_data: Dict[str, Any] = field(default_factory=dict)


class GeminiAdapter:
    """
    Adapter for Gemini 3 Flash via OpenAI-compatible API.
    Handles Akinator-style questioning for water quality inference.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize the Gemini adapter with API configuration."""
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL
        self.model = model or MODEL_NAME
        self._initialized = False
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> bool:
        """Initialize the HTTP client and verify API connectivity."""
        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
            
            # Test connectivity with a simple request
            try:
                response = await self._client.get("/models")
                if response.status_code in (200, 401, 403):
                    self._initialized = True
                    logger.info(f"GeminiAdapter initialized successfully. Model: {self.model}")
                    return True
            except httpx.ConnectError:
                logger.warning(f"Could not connect to {self.base_url}, will retry on use")
                self._initialized = True  # Allow lazy initialization
                return True

            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize GeminiAdapter: {e}")
            return False

    async def _make_api_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Make a chat completion request to the API."""
        if not self._client:
            await self.initialize()

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"API request error: {e}")
            return None

    def _build_system_prompt(self, state: AkinatorState) -> str:
        """Build the system prompt with detection context."""
        base_prompt = """You are HydroScan's Akinator AI, an intelligent water quality analysis assistant. Your role is to help identify water quality conditions through interactive questioning.

WATER QUALITY CATEGORIES:
- Clean: Clear, safe water with no visible contamination
- Dirty: Water with visible contamination, discoloration, or debris
- NotWater: Image doesn't contain water at all

DETECTION CONTEXT (from image analysis):
"""
        
        # Add detection context if available
        context_parts = []
        detection = state.detection_context
        
        if detection:
            if "detection_rate" in detection:
                context_parts.append(f"- Current detection rate: {detection['detection_rate']:.1%}")
            if "water_found" in detection:
                context_parts.append(f"- Water detected: {detection['water_found']}")
            if "objects_detected" in detection:
                context_parts.append(f"- Objects found: {', '.join(detection['objects_detected'])}")
            if "place_classification" in detection:
                context_parts.append(f"- Scene type: {detection['place_classification']}")
            if "confidence" in detection:
                context_parts.append(f"- Detection confidence: {detection['confidence']:.1%}")
        
        if state.user_description:
            context_parts.append(f"- USER DESCRIPTION: {state.user_description}")
        
        if context_parts:
            base_prompt += "\n".join(context_parts)
        else:
            base_prompt += "- No detection data available yet"

        base_prompt += """

QUESTIONING RULES:
1. Ask ONE clear, specific question at a time
2. Questions should be answerable with short sentences (not just yes/no)
3. Focus on details that help distinguish between Clean, Dirty, or NotWater
4. Consider the user's description and detection context
5. After max 10 questions, provide your final assessment

QUESTION FORMAT: Just ask the question directly, nothing else."""

        return base_prompt

    def _build_question_context(self, state: AkinatorState) -> List[Dict[str, str]]:
        """Build the conversation context for generating a question."""
        messages = [
            {"role": "system", "content": self._build_system_prompt(state)},
        ]
        
        # Add conversation history
        for entry in state.conversation_history:
            messages.append(entry)
        
        # Add instruction for next question
        if state.question_count == 0:
            messages.append({
                "role": "user",
                "content": f"Start the questioning. Round {state.question_count + 1}/{state.max_questions}. Ask your first question to help identify the water quality."
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Based on my answer, ask your next question. Round {state.question_count + 1}/{state.max_questions}."
            })
        
        return messages

    async def generate_question(
        self,
        session_id: str,
        detection_context: Dict[str, Any],
        user_description: Optional[str] = None,
    ) -> Question:
        """Generate the next question for the Akinator session."""
        # Create or get state
        state = AkinatorState(
            session_id=session_id,
            detection_context=detection_context,
            user_description=user_description,
        )

        messages = self._build_question_context(state)
        response = await self._make_api_request(messages, max_tokens=200, temperature=0.7)

        if response:
            question_text = response.strip()
        else:
            question_text = self._get_fallback_question(state)

        return Question(
            question_id=str(uuid.uuid4()),
            question_text=question_text,
            round_number=state.question_count + 1,
            context=json.dumps(detection_context) if detection_context else None,
        )

    async def process_answer(
        self,
        session_id: str,
        state: AkinatorState,
        answer: str,
    ) -> tuple[Optional[Question], Optional[InferenceResult]]:
        """
        Process user's answer and either generate next question or make final prediction.
        
        Returns:
            Tuple of (next_question, final_result) - one will be None
        """
        # Record the answer in conversation history
        # Find the last question from history or use a placeholder
        last_question = "What is the water condition?"
        for i in range(len(state.conversation_history) - 1, -1, -1):
            if state.conversation_history[i].get("role") == "assistant":
                last_question = state.conversation_history[i].get("content", last_question)
                break
        
        state.conversation_history.append({
            "role": "user",
            "content": f"Q: {last_question}\nMy answer: {answer}"
        })
        state.question_count += 1

        # Check if we should make a final prediction
        should_predict = (
            state.question_count >= state.max_questions or
            state.cancelled or
            self._should_early_predict(state)
        )

        if should_predict:
            result = await self._make_final_prediction(state)
            return None, result

        # Generate next question
        messages = self._build_question_context(state)
        response = await self._make_api_request(messages, max_tokens=200, temperature=0.7)

        if response:
            question_text = response.strip()
        else:
            question_text = self._get_fallback_question(state)

        next_question = Question(
            question_id=str(uuid.uuid4()),
            question_text=question_text,
            round_number=state.question_count + 1,
        )

        return next_question, None

    def _should_early_predict(self, state: AkinatorState) -> bool:
        """Determine if we have enough confidence to predict early."""
        # Simple heuristic: predict early if we have strong signals
        # This could be enhanced with actual confidence tracking
        return False  # For now, use all questions

    async def _make_final_prediction(self, state: AkinatorState) -> InferenceResult:
        """Generate the final water quality prediction."""
        system_prompt = self._build_system_prompt(state)
        
        messages = [
            {"role": "system", "content": system_prompt + """

Now provide your FINAL ASSESSMENT in this exact JSON format:
{
    "label": "Clean" | "Dirty" | "NotWater",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your conclusion"
}

Only output the JSON, nothing else."""},
        ]
        
        # Add full conversation history
        for entry in state.conversation_history:
            messages.append(entry)
        
        messages.append({
            "role": "user",
            "content": f"Based on all {state.question_count} rounds of questioning, provide your final assessment now."
        })

        response = await self._make_api_request(messages, max_tokens=300, temperature=0.3)

        if response:
            result = self._parse_final_response(response, state)
        else:
            result = self._get_fallback_prediction(state)

        return result

    def _parse_final_response(self, response: str, state: AkinatorState) -> InferenceResult:
        """Parse the API response into an InferenceResult."""
        try:
            # Try to extract JSON from the response
            json_match = response
            if "{" in response and "}" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_match = response[start:end]
            
            data = json.loads(json_match)
            
            label = data.get("label", "Unknown")
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "Based on the questioning")

            # Determine confidence level
            if confidence >= 0.9:
                level = ConfidenceLevel.VERY_HIGH
            elif confidence >= 0.7:
                level = ConfidenceLevel.HIGH
            elif confidence >= 0.5:
                level = ConfidenceLevel.MEDIUM
            else:
                level = ConfidenceLevel.LOW

            # If cancelled early, reduce confidence
            if state.cancelled:
                confidence = min(confidence * 0.8, 0.7)  # Cap at 70% for early cancellation
                reasoning += " (Confidence reduced due to early cancellation)"
                level = ConfidenceLevel.MEDIUM if level == ConfidenceLevel.HIGH else level

            return InferenceResult(
                predicted_label=label,
                confidence=confidence,
                confidence_level=level,
                reasoning=reasoning,
                question_count=state.question_count,
                detection_data=state.detection_context,
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse final response: {e}")
            return self._get_fallback_prediction(state)

    def _get_fallback_question(self, state: AkinatorState) -> str:
        """Get a fallback question if API fails."""
        fallback_questions = [
            "Can you describe the color of the water you see?",
            "Is there any visible debris or particles in the water?",
            "Does the water appear clear or murky?",
            "Can you see through the water to the bottom?",
            "Is there any foam or scum on the water surface?",
            "What is the surrounding environment like (natural, urban, indoor)?",
            "Is the water flowing or stagnant?",
            "Are there any plants or algae visible in or around the water?",
            "Does the water have any unusual coloration (green, brown, yellow)?",
            "Is there anyone using or interacting with the water?",
        ]
        
        idx = min(state.question_count, len(fallback_questions) - 1)
        return fallback_questions[idx]

    def _get_fallback_prediction(self, state: AkinatorState) -> InferenceResult:
        """Get a fallback prediction if API fails."""
        # Analyze conversation for hints
        conversation_text = " ".join([
            entry.get("content", "") for entry in state.conversation_history
        ]).lower()

        # Simple keyword-based prediction
        dirty_keywords = ["dirty", "murky", "brown", "green", "polluted", "debris", "trash", "foam", "algae"]
        clean_keywords = ["clear", "clean", "transparent", "blue", "fresh", "drinking"]
        notwater_keywords = ["no water", "not water", "dry", "land", "ground", "floor"]

        dirty_count = sum(1 for kw in dirty_keywords if kw in conversation_text)
        clean_count = sum(1 for kw in clean_keywords if kw in conversation_text)
        notwater_count = sum(1 for kw in notwater_keywords if kw in conversation_text)

        if notwater_count > max(dirty_count, clean_count):
            label = "NotWater"
            confidence = 0.5 + (notwater_count * 0.1)
        elif dirty_count > clean_count:
            label = "Dirty"
            confidence = 0.5 + (dirty_count * 0.1)
        else:
            label = "Clean"
            confidence = 0.5 + (clean_count * 0.1)

        confidence = min(confidence, 0.85)
        
        return InferenceResult(
            predicted_label=label,
            confidence=confidence,
            confidence_level=ConfidenceLevel.MEDIUM,
            reasoning="Fallback prediction based on conversation analysis",
            question_count=state.question_count,
            detection_data=state.detection_context,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False


# Module-level adapter instance (renamed for compatibility)
_adapter: Optional[GeminiAdapter] = None


async def get_qwen_adapter() -> GeminiAdapter:
    """Get or create the Gemini adapter instance (named for backward compatibility)."""
    global _adapter
    if _adapter is None:
        _adapter = GeminiAdapter()
        await _adapter.initialize()
    return _adapter


async def close_qwen_adapter() -> None:
    """Close the Gemini adapter (named for backward compatibility)."""
    global _adapter
    if _adapter:
        await _adapter.close()
        _adapter = None