"""
User Input Adapter for text-based water quality analysis.

This adapter analyzes user-provided text descriptions (smell, color, clarity, etc.)
to contribute to the overall water quality assessment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class UserInputAssessment:
    """Assessment result from user input analysis."""
    available: bool = False
    conclusion: str = ""
    score: float = 0.0
    confidence: float = 0.0
    rationale: Optional[str] = None
    model_name: Optional[str] = None
    reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary."""
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
    """
    Adapter for analyzing user-provided text descriptions.
    
    This adapter uses keyword matching and heuristic analysis to extract
    water quality indicators from user descriptions.
    """

    # Keywords that suggest water quality issues
    DIRTY_KEYWORDS = [
        "dirty", "kotor", "murky", "cloudy", "berwarna", "keruh",
        "smell", "bau", "odor", "berbau", "stink", "busuk",
        "brown", "coklat", "yellow", "kuning", "green", "hijau",
        "rust", "karat", "metallic", "logam", "iron", "besi",
        "contaminated", "terkontaminasi", "polluted", "tercemar",
        "unsafe", "tidak aman", "toxic", "beracun",
        "particles", "partikel", "sediment", "sedimen", "debris",
        "algae", "alga", "moss", "lumut", "slime", "lender",
        "sewage", "limbah", "waste", "sampah",
    ]
    
    CLEAN_KEYWORDS = [
        "clean", "bersih", "clear", "jernih", "transparent", "transparan",
        "fresh", "segar", "pure", "murni", "safe", "aman",
        "drinkable", "dapat diminum", "potable", "layak minum",
        "no smell", "tidak berbau", "odorless", "tidak berbau",
        "colorless", "tidak berwarna", "tasteless", "tidak berasa",
    ]
    
    # Severity modifiers
    SEVERE_KEYWORDS = [
        "very", "sangat", "extremely", "sangat", "strong", "kuat",
        "heavy", "berat", "severe", "parah", "dangerous", "berbahaya",
    ]

    def __init__(self, models_dir: Optional[str] = None):
        """Initialize the UserInput adapter."""
        self.models_dir = Path(models_dir) if models_dir else None
        self.available = True  # Always available (uses heuristics, no ML model required)
        self.status = "ready"
        self.model_name = "heuristic-keyword-v1"

    def analyze(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> UserInputAssessment:
        """
        Analyze user-provided text description for water quality indicators.
        
        Args:
            description: User's text description of the water
            context: Optional context including detection results, visual metrics, etc.
            
        Returns:
            UserInputAssessment with analysis results
        """
        if not description or not description.strip():
            return UserInputAssessment(
                available=True,
                conclusion="No user description provided",
                score=50.0,  # Neutral score
                confidence=0.0,
                rationale="No text input to analyze",
                model_name=self.model_name,
            )

        text = description.lower().strip()
        
        # Count keyword matches
        dirty_count = sum(1 for kw in self.DIRTY_KEYWORDS if kw in text)
        clean_count = sum(1 for kw in self.CLEAN_KEYWORDS if kw in text)
        severe_count = sum(1 for kw in self.SEVERE_KEYWORDS if kw in text)
        
        # Calculate base score (0-100, higher = cleaner)
        total_keywords = dirty_count + clean_count
        if total_keywords == 0:
            # No relevant keywords found
            base_score = 50.0
            confidence = 0.0
            conclusion = "No specific water quality indicators found in description"
            rationale = "The description did not contain recognizable water quality keywords"
        else:
            # Calculate score based on keyword ratio
            clean_ratio = clean_count / total_keywords
            base_score = clean_ratio * 100.0
            
            # Adjust confidence based on keyword count
            confidence = min(0.9, total_keywords * 0.15)
            
            # Adjust score for severity modifiers
            if severe_count > 0 and dirty_count > 0:
                base_score = max(0, base_score - severe_count * 10)
                confidence = min(0.95, confidence + 0.1)
            
            # Generate conclusion
            if base_score >= 70:
                conclusion = "Description suggests clean/safe water"
            elif base_score >= 50:
                conclusion = "Description suggests acceptable water quality"
            elif base_score >= 30:
                conclusion = "Description suggests potential water quality issues"
            else:
                conclusion = "Description suggests contaminated/unsafe water"
            
            rationale = f"Found {clean_count} clean indicators, {dirty_count} concern indicators, {severe_count} severity modifiers"
        
        # Incorporate context if available
        if context:
            base_scores = context.get("base_scores", {})
            if base_scores:
                # Blend with visual analysis scores if available
                visual_score = base_scores.get("potability_score", 50)
                # Weight: 30% user input, 70% visual analysis
                base_score = 0.3 * base_score + 0.7 * visual_score
                confidence = min(0.9, confidence + 0.2)
        
        return UserInputAssessment(
            available=True,
            conclusion=conclusion,
            score=round(base_score, 1),
            confidence=round(confidence, 2),
            rationale=rationale,
            model_name=self.model_name,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            "available": self.available,
            "status": self.status,
            "model_name": self.model_name,
            "models_dir": str(self.models_dir) if self.models_dir else None,
        }


# Module-level functions for compatibility
_adapter_instance: Optional[UserInputAdapter] = None


def get_adapter(models_dir: Optional[str] = None) -> UserInputAdapter:
    """Get or create the singleton adapter instance."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = UserInputAdapter(models_dir=models_dir)
    return _adapter_instance