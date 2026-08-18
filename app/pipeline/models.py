from dataclasses import dataclass, field
from typing import Any, Optional

from app.quality.models import QualityReport
from app.imputation.models import ImputationRecommendation
from app.cleaning.models import CleaningRecommendation


@dataclass
class PipelineResult:

    # ==========================================
    # DATA
    # ==========================================

    original_dataframe: Any

    cleaned_dataframe: Any

    # ==========================================
    # PROFILE
    # ==========================================

    dataset_profile: Any = None

    # ==========================================
    # MISSING ANALYSIS
    # ==========================================

    missing_analysis: list[dict] = field(
        default_factory=list
    )

    missing_row_analysis: list[dict] = field(
        default_factory=list
    )

    missing_patterns: list[dict] = field(
        default_factory=list
    )

    missing_correlation: Any = None

    # ==========================================
    # ANOMALIES
    # ==========================================

    rule_anomalies: list = field(
        default_factory=list
    )

    statistical_anomalies: list = field(
        default_factory=list
    )

    anomalies: list = field(
        default_factory=list
    )

    # ==========================================
    # CLEANING
    # ==========================================

    cleaning_recommendations: list[
        CleaningRecommendation
    ] = field(
        default_factory=list
    )

    cleaning_logs: list[dict] = field(
        default_factory=list
    )

    # ==========================================
    # IMPUTATION
    # ==========================================

    imputation_recommendations: list[
        ImputationRecommendation
    ] = field(
        default_factory=list
    )

    imputation_logs: list[dict] = field(
        default_factory=list
    )
    
    quality_comparison: Any = None

    # ==========================================
    # QUALITY
    # ==========================================

    quality_report: Optional[
        QualityReport
    ] = None

    # ==========================================
    # BACKWARD COMPATIBILITY
    # ==========================================

    @property
    def dataframe(self):
        """
        Backward-compatible alias.

        Older code/tests may access:
            result.dataframe

        The pipeline now distinguishes:
            result.original_dataframe
            result.cleaned_dataframe

        Therefore dataframe refers to the
        final cleaned dataframe.
        """
        return self.cleaned_dataframe
    
    # ==========================================
    # BACKWARD COMPATIBILITY
    # ==========================================

    @property
    def dataframe(self):
        """
        Backward-compatible alias for the
        final cleaned dataframe.
        """
        return self.cleaned_dataframe

    @property
    def cleaning_results(self):
        """
        Backward-compatible alias for cleaning logs.
        """
        return self.cleaning_logs

    @property
    def imputation_results(self):
        """
        Backward-compatible alias for imputation logs.
        """
        return self.imputation_logs