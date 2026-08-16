from dataclasses import dataclass


@dataclass
class CleaningRecommendation:

    row_index: int

    column: str

    value: object

    anomaly_types: list[str]

    action: str

    confidence: float

    reason: str

    requires_review: bool