from dataclasses import dataclass, field
from typing import Any


@dataclass
class HybridAnomalyRecord:

    # ==========================================================
    # LOCATION
    # ==========================================================

    row_index: int

    column: str

    value: Any

    # ==========================================================
    # DETECTOR AGREEMENT
    # ==========================================================

    detectors: list[str] = field(
        default_factory=list
    )

    detector_count: int = 0

    total_detectors: int = 0

    agreement_ratio: float = 0.0

    # ==========================================================
    # DETECTOR TYPE
    # ==========================================================

    detection_scope: str = "column"

    # column:
    #     IQR / Z-score / semantic
    #
    # multivariate:
    #     Isolation Forest / LOF / DBSCAN

    # ==========================================================
    # CONFIDENCE
    # ==========================================================

    confidence: float = 0.0

    # ==========================================================
    # FINAL CLASSIFICATION
    # ==========================================================

    anomaly_type: str = "outlier"

    severity: str = "low"

    # ==========================================================
    # EXPLANATION
    # ==========================================================

    reason: str = ""