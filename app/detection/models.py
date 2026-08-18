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

    method: str
    
    severity: str

    reason: str