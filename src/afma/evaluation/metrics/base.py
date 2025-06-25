from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Result of an evaluation metric."""
    metric_name: str
    score: float
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None 