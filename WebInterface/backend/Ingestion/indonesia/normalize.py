from typing import Dict, Any


def normalize_external(raw: Dict[str, Any] | None) -> Dict[str, Any]:
    """Normalize external data into a common schema. Placeholder passthrough."""
    return raw or {}
