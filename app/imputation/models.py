from dataclasses import dataclass
from typing import Optional


@dataclass
class ImputationRecommendation:

    column: str

    strategy: str

    confidence: float

    missing_count: int

    missing_ratio: float

    reason: str

    requires_review: bool = False

    fill_value: Optional[object] = None