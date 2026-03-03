"""
Akinator-style water quality inference API endpoints.

This module provides interactive back-and-forth questioning to determine
water quality, similar to the Akinator game. The AI asks questions and
narrows down the water quality assessment based on user answers.
"""

from __future__ import annotations

import uuid
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import the QwenAdapter for local LLM inference
from WebInterface.backend.Adapters.QwenAdapter import (
    QwenAdapter,
    AkinatorState,
    Question,
    InferenceResult,
)

router = APIRouter(prefix="/api/akinator", tags=["akinator"])

# In-memory session storage (for production, use Redis or database)
_sessions: Dict[str, AkinatorState] = {}
_adapter: Optional[QwenAdapter] = None


def get_adapter() -> QwenAdapter:
    """Get or create the QwenAdapter singleton."""
    global _adapter
    if _adapter is None:
        _adapter = QwenAdapter()
    return _adapter


class StartSessionRequest(BaseModel):
    """Request to start a new Akinator session."""
    initial_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Initial context from visual analysis (scores, detections, etc.)"
    )
    analysis_id: Optional[str] = Field(
        default=None,
        description="Optional analysis ID to link this session to"
    )


class AnswerRequest(BaseModel):
    """Request to submit an answer to the current question."""
    session_id: str = Field(..., description="The session ID")
    question_id: str = Field(..., description="The question ID being answered")
    answer: str = Field(..., description="The user's answer (yes/no/unsure or text)")


class SessionResponse(BaseModel):
    """Response containing session state and current question."""
    session_id: str
    round_number: int
    max_rounds: int
    question: Optional[Dict[str, Any]] = None
    inference: Optional[Dict[str, Any]] = None
    status: str  # "questioning", "confident", "max_rounds", "error"
    message: Optional[str] = None


class InferenceResponse(BaseModel):
    """Response containing the final inference result."""
    session_id: str
    inference: Dict[str, Any]
    confidence: float
    rounds_used: int
    status: str


@router.post("/start", response_model=SessionResponse)
async def start_session(request: StartSessionRequest) -> SessionResponse:
    """
    Start a new Akinator session.
    
    This initializes a new questioning session. If initial_context is provided
    (from visual analysis), the AI will use that to inform its questions.
    """
    session_id = str(uuid.uuid4())
    adapter = get_adapter()
    
    # Create initial state
    state = AkinatorState(
        session_id=session_id,
        analysis_id=request.analysis_id,
        visual_context=request.initial_context or {},
        answers=[],
        confidence_history=[],
    )
    
    # Store session
    _sessions[session_id] = state
    
    try:
        # Generate first question
        question = await adapter.generate_question(state, request.initial_context)
        
        if question is None:
            # Model unavailable, use fallback
            state.current_question = adapter.get_fallback_question(0)
        else:
            state.current_question = question
        
        return SessionResponse(
            session_id=session_id,
            round_number=1,
            max_rounds=10,
            question={
                "id": state.current_question.id,
                "text": state.current_question.text,
                "type": state.current_question.question_type,
                "options": state.current_question.options,
            } if state.current_question else None,
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
    """
    state = _sessions.get(request.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if state.current_question is None or state.current_question.id != request.question_id:
        raise HTTPException(status_code=400, detail="Invalid question ID for this session")
    
    adapter = get_adapter()
    
    try:
        # Process the answer
        next_question, inference = await adapter.process_answer(
            state, request.answer, request.question_id
        )
        
        # Check if we have an inference
        if inference is not None:
            # Session complete - return inference
            _sessions.pop(request.session_id, None)  # Clean up
            
            return SessionResponse(
                session_id=request.session_id,
                round_number=state.round_number,
                max_rounds=10,
                question=None,
                inference={
                    "water_quality": inference.water_quality,
                    "confidence": inference.confidence,
                    "reasoning": inference.reasoning,
                    "recommendations": inference.recommendations,
                    "contaminants": inference.contaminants,
                },
                status="confident" if inference.confidence >= 0.85 else "max_rounds",
                message="Analysis complete! Here's my assessment.",
            )
        
        # Need more questions
        if next_question is not None:
            state.current_question = next_question
            
            return SessionResponse(
                session_id=request.session_id,
                round_number=state.round_number,
                max_rounds=10,
                question={
                    "id": next_question.id,
                    "text": next_question.text,
                    "type": next_question.question_type,
                    "options": next_question.options,
                },
                inference=None,
                status="questioning",
                message=f"Round {state.round_number} of 10",
            )
        
        # No more questions but no inference - use fallback
        # This shouldn't happen normally, but handle gracefully
        _sessions.pop(request.session_id, None)
        
        return SessionResponse(
            session_id=request.session_id,
            round_number=state.round_number,
            max_rounds=10,
            question=None,
            inference={
                "water_quality": "Unknown",
                "confidence": 0.0,
                "reasoning": "Unable to determine water quality from the provided answers.",
                "recommendations": ["Consider re-analyzing with clearer answers."],
                "contaminants": [],
            },
            status="error",
            message="Could not determine water quality.",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process answer: {str(e)}")


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    """Get the current state of an Akinator session."""
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        session_id=session_id,
        round_number=state.round_number,
        max_rounds=10,
        question={
            "id": state.current_question.id,
            "text": state.current_question.text,
            "type": state.current_question.question_type,
            "options": state.current_question.options,
        } if state.current_question else None,
        inference=None,
        status="questioning",
        message=f"Session in progress. Round {state.round_number} of 10.",
    )


@router.delete("/session/{session_id}")
async def end_session(session_id: str) -> JSONResponse:
    """End and clean up an Akinator session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return JSONResponse({"message": "Session ended successfully"})
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check the health status of the Akinator system."""
    adapter = get_adapter()
    
    return {
        "status": "healthy",
        "model_available": adapter.is_available(),
        "device": adapter.device if hasattr(adapter, 'device') else "unknown",
        "active_sessions": len(_sessions),
    }


# Cleanup task for expired sessions (optional, for production)
async def cleanup_expired_sessions(max_age_minutes: int = 30):
    """Remove sessions that have been inactive for too long."""
    # This would be called periodically by a background task
    # For now, sessions are cleaned up when they complete
    pass