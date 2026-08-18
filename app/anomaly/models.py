from dataclasses import dataclass
from typing import Any


@dataclass
class AnomalyRecord:

    row_index: int

    column: str

    value: Any

    anomaly_type: str

    detector: str

    score: float

    severity: str

    method: str

    reason: str