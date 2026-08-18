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
    
@dataclass
class QualitySnapshot:

    # ==========================================
    # DATASET
    # ==========================================

    rows: int
    columns: int
    total_cells: int

    # ==========================================
    # MISSING
    # ==========================================

    missing_cells: int
    missing_ratio: float

    # ==========================================
    # ANOMALIES
    # ==========================================

    anomaly_count: int

    # ==========================================
    # QUALITY
    # ==========================================

    quality_score: float


@dataclass
class QualityComparison:

    # ==========================================
    # BEFORE
    # ==========================================

    before: QualitySnapshot

    # ==========================================
    # AFTER
    # ==========================================

    after: QualitySnapshot

    # ==========================================
    # IMPROVEMENT
    # ==========================================

    missing_reduction: int

    missing_ratio_reduction: float

    anomaly_reduction: int

    quality_score_improvement: float

    quality_improved: bool