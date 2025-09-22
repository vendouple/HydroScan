from typing import Dict, Any


def score(
    visual: Dict[str, float], external: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Compute potability and confidence from visual metrics and optional external data (stub)."""
    potability = 50.0
    confidence = 25.0
    band = "Clean for daily use, not drinkable"
    return {
        "potability_score": potability,
        "confidence_score": confidence,
        "band_label": band,
        "components": {"visual": visual, "external": external or {}},
    }
