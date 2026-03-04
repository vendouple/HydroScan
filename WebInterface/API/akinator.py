"""
Akinator-style water quality inference API endpoints.

This module provides interactive back-and-forth questioning to determine
water quality, similar to the Akinator game. The AI asks questions and
narrows down the water quality assessment based on user answers.

Uses Gemini 3 Flash via OpenAI-compatible API for inference.
"""

from __future__ import annotations

import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import the GeminiAdapter (named QwenAdapter for backward compatibility)
from WebInterface.backend.Adapters.QwenAdapter import (
    GeminiAdapter,
    AkinatorState,
    Question,
    InferenceResult,
    get_qwen_adapter,
    close_qwen_adapter,
)

router = APIRouter(prefix="/api/akinator", tags=["akinator"])

# In-memory session storage (for production, use Redis or database)
_sessions: Dict[str, AkinatorState] = {}


class StartSessionRequest(BaseModel):
    """Request to start a new Akinator session."""
    detection_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detection context from visual analysis (detection_rate, water_found, objects, etc.)"
    )
    user_description: Optional[str] = Field(
        default=None,
        description="User's text description of the water/image"
    )
    analysis_id: Optional[str] = Field(
        default=None,
        description="Optional analysis ID to link this session to"
    )


class AnswerRequest(BaseModel):
    """Request to submit an answer to the current question."""
    session_id: str = Field(..., description="The session ID")
    answer: str = Field(..., description="The user's answer (short sentence or phrase)")


class CancelRequest(BaseModel):
    """Request to cancel the session early and get results."""
    session_id: str = Field(..., description="The session ID")


class SessionResponse(BaseModel):
    """Response containing session state and current question."""
    session_id: str
    round_number: int
    max_rounds: int
    question: Optional[Dict[str, Any]] = None
    inference: Optional[Dict[str, Any]] = None
    status: str  # "questioning", "cancelled", "confident", "max_rounds", "error"
    message: Optional[str] = None
    warning: Optional[str] = None


@router.post("/start", response_model=SessionResponse)
async def start_session(request: StartSessionRequest) -> SessionResponse:
    """
    Start a new Akinator session.
    
    This initializes a new questioning session. If detection_context is provided
    (from visual analysis), the AI will use that to inform its questions.
    The user_description is also used to help guide the questioning.
    """
    session_id = str(uuid.uuid4())
    adapter = await get_qwen_adapter()
    
    # Create initial state
    state = AkinatorState(
        session_id=session_id,
        detection_context=request.detection_context or {},
        user_description=request.user_description,
    )
    
    # Store session
    _sessions[session_id] = state
    
    try:
        # Generate first question
        question = await adapter.generate_question(
            session_id=session_id,
            detection_context=request.detection_context or {},
            user_description=request.user_description,
        )
        
        if question is None:
            # Model unavailable, use fallback
            state.question_count = 0
            question = Question(
                question_id=str(uuid.uuid4()),
                question_text="Can you describe what you see in the image related to water?",
                round_number=1,
            )
        
        return SessionResponse(
            session_id=session_id,
            round_number=question.round_number,
            max_rounds=state.max_questions,
            question={
                "id": question.question_id,
                "text": question.question_text,
                "round": question.round_number,
            },
            inference=None,
            status="questioning",
            message="Session started. Answer the question to continue.",
        )
        
    except Exception as e:
        # Clean up session on error
        _sessions.pop(session_id, None)
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")


@router.post("/answer", response_model=SessionResponse)
async def submit_answer(request: AnswerRequest) -> SessionResponse:
    """
    Submit an answer to the current question.
    
    The AI will process the answer and either:
    1. Ask another question (if not confident yet)
    2. Return an inference (if confident or max rounds reached)
    
    Answers can be short sentences or phrases - not just yes/no/maybe.
    """
    state = _sessions.get(request.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    adapter = await get_qwen_adapter()
    
    try:
        # Process the answer
        next_question, inference = await adapter.process_answer(
            session_id=request.session_id,
            state=state,
            answer=request.answer,
        )
        
        # Check if we have an inference
        if inference is not None:
            # Session complete - return inference
            _sessions.pop(request.session_id, None)  # Clean up
            
            status = "confident" if inference.confidence >= 0.85 else "max_rounds"
            if state.cancelled:
                status = "cancelled"
            
            return SessionResponse(
                session_id=request.session_id,
                round_number=inference.question_count,
                max_rounds=state.max_questions,
                question=None,
                inference={
                    "label": inference.predicted_label,
                    "confidence": inference.confidence,
                    "confidence_level": inference.confidence_level.value,
                    "reasoning": inference.reasoning,
                    "detection_data": inference.detection_data,
                },
                status=status,
                message="Analysis complete! Here's my assessment.",
                warning="Results may be less accurate due to early cancellation." if state.cancelled else None,
            )
        
        # Need more questions
        if next_question is not None:
            return SessionResponse(
                session_id=request.session_id,
                round_number=next_question.round_number,
                max_rounds=state.max_questions,
                question={
                    "id": next_question.question_id,
                    "text": next_question.question_text,
                    "round": next_question.round_number,
                },
                inference=None,
                status="questioning",
                message=f"Round {next_question.round_number} of {state.max_questions}",
            )
        
        # No more questions but no inference - use fallback
        _sessions.pop(request.session_id, None)
        
        return SessionResponse(
            session_id=request.session_id,
            round_number=state.question_count,
            max_rounds=state.max_questions,
            question=None,
            inference={
                "label": "Unknown",
                "confidence": 0.0,
                "confidence_level": "low",
                "reasoning": "Unable to determine water quality from the provided answers.",
                "detection_data": state.detection_context,
            },
            status="error",
            message="Could not determine water quality.",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process answer: {str(e)}")


@router.post("/cancel", response_model=SessionResponse)
async def cancel_session(request: CancelRequest) -> SessionResponse:
    """
    Cancel the session early and get immediate results with a warning.
    
    This allows users to end the questioning before all rounds are complete.
    Results will be marked as potentially less accurate.
    """
    state = _sessions.get(request.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Mark as cancelled
    state.cancelled = True
    
    adapter = await get_qwen_adapter()
    
    try:
        # Force a final prediction
        inference = await adapter._make_final_prediction(state)
        
        # Clean up session
        _sessions.pop(request.session_id, None)
        
        return SessionResponse(
            session_id=request.session_id,
            round_number=inference.question_count,
            max_rounds=state.max_questions,
            question=None,
            inference={
                "label": inference.predicted_label,
                "confidence": inference.confidence,
                "confidence_level": inference.confidence_level.value,
                "reasoning": inference.reasoning,
                "detection_data": inference.detection_data,
            },
            status="cancelled",
            message="Session cancelled. Results provided based on limited questioning.",
            warning="WARNING: Results may be less accurate due to early cancellation. For best results, complete all questioning rounds.",
        )
        
    except Exception as e:
        _sessions.pop(request.session_id, None)
        raise HTTPException(status_code=500, detail=f"Failed to cancel session: {str(e)}")


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    """Get the current state of an Akinator session."""
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Build response - we don't store the current question in state anymore
    return SessionResponse(
        session_id=session_id,
        round_number=state.question_count,
        max_rounds=state.max_questions,
        question=None,  # Question is generated on-demand
        inference=None,
        status="questioning",
        message=f"Session in progress. Round {state.question_count} of {state.max_questions} completed.",
    )


@router.delete("/session/{session_id}")
async def end_session(session_id: str) -> JSONResponse:
    """End and clean up an Akinator session without getting results."""
    if session_id in _sessions:
        del _sessions[session_id]
        return JSONResponse({"message": "Session ended successfully"})
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check the health status of the Akinator system."""
    adapter = await get_qwen_adapter()
    
    return {
        "status": "healthy",
        "model": "gemini-3-flash-preview",
        "api_endpoint": "helixmind.online/v1",
        "adapter_initialized": adapter._initialized,
        "active_sessions": len(_sessions),
    }


@router.on_event("shutdown")
async def shutdown_event():
    """Clean up adapter on shutdown."""
    await close_qwen_adapter()