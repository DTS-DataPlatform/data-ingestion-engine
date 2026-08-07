from dataclasses import dataclass, field


@dataclass
class QualityReport:

    # ==========================================
    # DATASET
    # ==========================================

    rows: int
    columns: int
    total_cells: int

    # ==========================================
    # MISSING VALUES
    # ==========================================

    missing_cells: int
    missing_ratio: float

    # ==========================================
    # QUALITY
    # ==========================================

    quality_score: float

    # ==========================================
    # ANOMALIES
    # ==========================================

    total_anomalies: int

    # ==========================================
    # CLEANING
    # ==========================================

    cleaned_count: int
    review_count: int
    skipped_count: int

    # ==========================================
    # OPTIONAL DETAILS
    # ==========================================

    severity_counts: dict[str, int] = field(
        default_factory=dict
    )

    issue_counts: dict[str, int] = field(
        default_factory=dict
    )

    column_issues: dict[str, dict] = field(
        default_factory=dict
    )