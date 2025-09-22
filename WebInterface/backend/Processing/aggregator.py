from typing import Dict, Any, List


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-frame/variant outputs into a summary. Placeholder passthrough."""
    return {"frames": results}
