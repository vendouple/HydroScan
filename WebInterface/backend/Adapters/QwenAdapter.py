"""
QwenAdapter - Local LLM integration for Akinator-style water quality inference.

This module provides integration with Qwen3.5-9B (or similar) local LLM for
intelligent questioning and inference about water quality.
"""

from __future__ import annotations

import os
import json
import logging
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for Akinator decisions."""
    LOW = "low"       # < 0.4
    MEDIUM = "medium" # 0.4 - 0.6
    HIGH = "high"     # 0.6 - 0.85
    VERY_HIGH = "very_high"  # > 0.85


@dataclass
class AkinatorState:
    """State for Akinator questioning session."""
    session_id: str
    round_number: int = 0
    max_rounds: int = 10
    questions_asked: List[Dict[str, Any]] = field(default_factory=list)
    answers_received: List[Dict[str, Any]] = field(default_factory=list)
    current_confidence: float = 0.0
    current_prediction: Optional[Dict[str, Any]] = None
    visual_context: Optional[Dict[str, Any]] = None
    is_complete: bool = False
    final_result: Optional[Dict[str, Any]] = None


@dataclass
class Question:
    """Represents a question from the Akinator system."""
    question_id: str
    text: str
    category: str  # smell, color, clarity, source, usage, etc.
    options: List[str] = field(default_factory=lambda: ["yes", "no", "maybe", "unsure"])
    importance: float = 1.0  # How important this question is for inference


@dataclass
class InferenceResult:
    """Result of Akinator inference."""
    quality_score: float
    confidence: float
    quality_label: str
    reasoning: str
    detected_issues: List[str]
    recommendations: List[str]
    is_final: bool


class QwenAdapter:
    """
    Adapter for Qwen3.5-9B local LLM integration.
    
    Supports GPU/CUDA with CPU fallback for water quality inference
    through Akinator-style questioning.
    """
    
    # System prompt for water quality Akinator
    SYSTEM_PROMPT = """You are an expert water quality analyst AI assistant. Your role is to help determine water quality through a series of targeted questions, similar to the Akinator game.

You will be given visual analysis results from computer vision models and should ask clarifying questions to improve the accuracy of your assessment.

Your goals:
1. Ask targeted questions about observable water characteristics
2. Gather information efficiently (max 10 rounds)
3. Provide a confidence score (0-100%) for your assessment
4. Stop early if you're confident (>85%) before reaching 10 rounds

Question categories to consider:
- Smell: Any unusual odors (chlorine, sulfur, earthy, chemical, etc.)
- Color: Water coloration (clear, yellow, brown, green, etc.)
- Clarity: Turbidity, particles visible, sediment
- Source: Where the water comes from (tap, well, river, rain, etc.)
- Usage: Intended use (drinking, cooking, bathing, irrigation)
- Location: Geographic context if relevant
- Time: How long stored, seasonal factors

Always respond in JSON format with:
{
    "question": "your question text",
    "category": "category name",
    "options": ["yes", "no", "maybe", "unsure"],
    "importance": 0.0-1.0,
    "current_confidence": 0-100,
    "reasoning": "brief explanation of why you're asking this"
}

If you have enough information to make a final assessment, respond with:
{
    "is_final": true,
    "quality_score": 0-100,
    "confidence": 0-100,
    "quality_label": "Excellent/Good/Moderate/Poor/Critical",
    "reasoning": "detailed explanation",
    "detected_issues": ["issue1", "issue2"],
    "recommendations": ["rec1", "rec2"]
}"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the Qwen adapter.
        
        Args:
            model_path: Path to the Qwen model. If None, uses default.
            device: Device to use ('cuda', 'cpu', or 'auto' for auto-detect).
        """
        self.model = None
        self.tokenizer = None
        self.device = self._detect_device() if device is None else device
        self.model_path = model_path
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=1)
        
    def _detect_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return "cuda"
        logger.info("CUDA not available, using CPU")
        return "cpu"
    
    def _get_model_path(self) -> str:
        """Get the model path, checking environment variables."""
        if self.model_path:
            return self.model_path
        
        # Check environment variable
        env_path = os.environ.get("QWEN_MODEL_PATH")
        if env_path:
            return env_path
        
        # Default paths to check
        default_paths = [
            "./models/Qwen2.5-7B-Instruct",
            "./models/Qwen3.5-9B",
            "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct",
        ]
        
        for path in default_paths:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                return expanded
        
        # Return HuggingFace model ID for auto-download
        return "Qwen/Qwen2.5-7B-Instruct"
    
    async def initialize(self) -> bool:
        """
        Initialize the model asynchronously.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        if self._initialized:
            return True
            
        try:
            # Run initialization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(self._executor, self._init_model)
            self._initialized = success
            return success
        except Exception as e:
            logger.error(f"Failed to initialize Qwen model: {e}")
            return False
    
    def _init_model(self) -> bool:
        """Initialize the model (runs in thread pool)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_path = self._get_model_path()
            logger.info(f"Loading Qwen model from: {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Load model with appropriate settings
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            self.model.eval()
            logger.info(f"Qwen model loaded successfully on {self.device}")
            return True
            
        except ImportError as e:
            logger.error(f"Transformers library not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if the adapter is ready for inference."""
        return self._initialized and self.model is not None and self.tokenizer is not None
    
    async def generate_question(
        self,
        state: AkinatorState,
        visual_context: Optional[Dict[str, Any]] = None
    ) -> Question:
        """
        Generate the next question based on current state.
        
        Args:
            state: Current Akinator session state.
            visual_context: Visual analysis results from CV models.
            
        Returns:
            The next question to ask.
        """
        if not self.is_ready():
            # Return a default question if model not ready
            return self._get_fallback_question(state)
        
        # Build context for the model
        context = self._build_context(state, visual_context)
        
        # Generate response
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self._executor,
            lambda: self._generate_response(context)
        )
        
        # Parse response into Question
        return self._parse_question_response(response, state.round_number)
    
    async def process_answer(
        self,
        state: AkinatorState,
        answer: str,
        question_id: str
    ) -> Tuple[Optional[Question], Optional[InferenceResult]]:
        """
        Process a user's answer and determine next action.
        
        Args:
            state: Current Akinator session state.
            answer: User's answer to the last question.
            question_id: ID of the question being answered.
            
        Returns:
            Tuple of (next_question or None, inference_result or None).
            If inference_result is not None, the session is complete.
        """
        # Record the answer
        state.answers_received.append({
            "question_id": question_id,
            "answer": answer,
            "round": state.round_number
        })
        
        # Check if we should make a final prediction
        if state.current_confidence >= 0.85 or state.round_number >= state.max_rounds - 1:
            result = await self._make_final_prediction(state)
            state.is_complete = True
            state.final_result = result.__dict__
            return None, result
        
        # Generate next question
        next_question = await self.generate_question(state)
        state.round_number += 1
        
        return next_question, None
    
    async def _make_final_prediction(self, state: AkinatorState) -> InferenceResult:
        """Make the final quality prediction."""
        if not self.is_ready():
            return self._get_fallback_prediction(state)
        
        context = self._build_context(state, None)
        context += "\n\nBased on all the information gathered, provide your final assessment."
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self._executor,
            lambda: self._generate_response(context, max_tokens=500)
        )
        
        return self._parse_final_response(response)
    
    def _build_context(
        self,
        state: AkinatorState,
        visual_context: Optional[Dict[str, Any]]
    ) -> str:
        """Build the context string for the model."""
        context_parts = [self.SYSTEM_PROMPT]
        
        # Add visual context if available
        if visual_context:
            context_parts.append("\n\nVisual Analysis Results:")
            context_parts.append(json.dumps(visual_context, indent=2))
        
        # Add conversation history
        if state.questions_asked or state.answers_received:
            context_parts.append("\n\nConversation History:")
            for i, q in enumerate(state.questions_asked):
                context_parts.append(f"\nQ{i+1}: {q.get('text', '')}")
                if i < len(state.answers_received):
                    a = state.answers_received[i]
                    context_parts.append(f"A{i+1}: {a.get('answer', '')}")
        
        # Add current state
        context_parts.append(f"\n\nCurrent Round: {state.round_number + 1}/{state.max_rounds}")
        context_parts.append(f"Current Confidence: {state.current_confidence * 100:.1f}%")
        
        return "\n".join(context_parts)
    
    def _generate_response(
        self,
        context: str,
        max_tokens: int = 300
    ) -> str:
        """Generate a response from the model."""
        try:
            inputs = self.tokenizer(context, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def _parse_question_response(
        self,
        response: str,
        round_num: int
    ) -> Question:
        """Parse model response into a Question object."""
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                return Question(
                    question_id=f"q_{round_num}_{hash(response) % 10000}",
                    text=data.get("question", "Is the water clear?"),
                    category=data.get("category", "clarity"),
                    options=data.get("options", ["yes", "no", "maybe", "unsure"]),
                    importance=float(data.get("importance", 1.0))
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse question response: {e}")
        
        # Return default question if parsing fails
        return self._get_fallback_question_from_response(response, round_num)
    
    def _parse_final_response(self, response: str) -> InferenceResult:
        """Parse model response into an InferenceResult."""
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                return InferenceResult(
                    quality_score=float(data.get("quality_score", 50)),
                    confidence=float(data.get("confidence", 50)) / 100,
                    quality_label=data.get("quality_label", "Moderate"),
                    reasoning=data.get("reasoning", "Based on available information."),
                    detected_issues=data.get("detected_issues", []),
                    recommendations=data.get("recommendations", []),
                    is_final=True
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse final response: {e}")
        
        return InferenceResult(
            quality_score=50,
            confidence=0.5,
            quality_label="Moderate",
            reasoning="Unable to determine precise quality from available information.",
            detected_issues=[],
            recommendations=["Consider professional testing for accurate results."],
            is_final=True
        )
    
    def _get_fallback_question(self, state: AkinatorState) -> Question:
        """Get a fallback question when model is unavailable."""
        # Predefined questions based on round number
        fallback_questions = [
            Question("fq_0", "Does the water have any unusual smell?", "smell"),
            Question("fq_1", "Is the water clear or does it appear cloudy?", "clarity"),
            Question("fq_2", "What color is the water?", "color", 
                    ["clear", "yellowish", "brown", "green", "other"]),
            Question("fq_3", "What is the source of this water?", "source",
                    ["tap", "well", "river", "rain", "bottled", "other"]),
            Question("fq_4", "Is this water intended for drinking?", "usage"),
            Question("fq_5", "Have you noticed any sediment or particles in the water?", "clarity"),
            Question("fq_6", "How long has this water been stored?", "time"),
            Question("fq_7", "Is there any visible film or scum on the surface?", "appearance"),
            Question("fq_8", "Does the water taste normal if you've tried it?", "taste",
                    ["yes", "no", "haven't tried", "unusual"]),
            Question("fq_9", "Are there any known water quality issues in your area?", "location"),
        ]
        
        idx = min(state.round_number, len(fallback_questions) - 1)
        return fallback_questions[idx]
    
    def _get_fallback_question_from_response(
        self,
        response: str,
        round_num: int
    ) -> Question:
        """Create a question from unstructured response text."""
        # Extract any question-like sentence
        sentences = response.replace("?", "?|").split("|")
        for s in sentences:
            s = s.strip()
            if s and (s.endswith("?") or "ask" in s.lower() or "question" in s.lower()):
                return Question(
                    question_id=f"fq_{round_num}",
                    text=s if s.endswith("?") else s + "?",
                    category="general"
                )
        
        return self._get_fallback_question(AkinatorState(session_id="fallback"))
    
    def _get_fallback_prediction(self, state: AkinatorState) -> InferenceResult:
        """Get a fallback prediction when model is unavailable."""
        # Simple heuristic based on answers
        score = 70  # Default to good quality
        issues = []
        
        for answer in state.answers_received:
            ans = answer.get("answer", "").lower()
            if ans in ["no", "unsure"]:
                score -= 5
            if "smell" in str(state.questions_asked) and ans == "yes":
                score -= 15
                issues.append("Unusual odor detected")
            if "cloudy" in ans or "brown" in ans or "yellow" in ans:
                score -= 10
                issues.append("Water discoloration")
        
        score = max(0, min(100, score))
        
        if score >= 80:
            label = "Good"
        elif score >= 60:
            label = "Moderate"
        else:
            label = "Poor"
        
        return InferenceResult(
            quality_score=score,
            confidence=0.5,
            quality_label=label,
            reasoning="Assessment based on user responses.",
            detected_issues=issues,
            recommendations=["Consider professional testing for confirmation."],
            is_final=True
        )
    
    async def close(self):
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
        
        if self.model:
            del self.model
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._initialized = False


# Singleton instance
_qwen_adapter: Optional[QwenAdapter] = None


async def get_qwen_adapter() -> QwenAdapter:
    """Get or create the Qwen adapter singleton."""
    global _qwen_adapter
    
    if _qwen_adapter is None:
        _qwen_adapter = QwenAdapter()
        await _qwen_adapter.initialize()
    
    return _qwen_adapter


async def close_qwen_adapter():
    """Close the Qwen adapter singleton."""
    global _qwen_adapter
    
    if _qwen_adapter:
        await _qwen_adapter.close()
        _qwen_adapter = None