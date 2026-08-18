from dataclasses import dataclass
from typing import Any


@dataclass
class ColumnProfile:

    # =========================
    # Basic information
    # =========================

    name: str
    dtype: str
    semantic_type: str | None

    missing_count: int
    missing_ratio: float

    unique_count: int
    unique_ratio: float

    numeric_ratio: float = 0.0

    # =========================
    # Numerical profiling
    # =========================

    min: Any = None
    max: Any = None

    mean: float | None = None
    median: float | None = None
    std: float | None = None

    quantiles: dict | None = None

    distribution: dict | None = None

    # =========================
    # Pattern detection
    # =========================

    pattern: dict | None = None

    # =========================
    # Semantic detection
    # =========================

    semantic_confidence: float = 0.0

    semantic_evidence: list[str] | None = None


@dataclass
class DatasetProfile:

    rows: int
    columns: int

    column_profiles: list[ColumnProfile]
    
    correlation: dict | None = None