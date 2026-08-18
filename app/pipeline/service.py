import pandas as pd

from app.core.table import UnifiedTable

from app.profiling.dataset_profiler import (
    profile_dataset,
)

from app.semantic.semantic_detector import (
    detect_semantic_types,
)

from app.imputation.analyzer import (
    analyze_missing_values,
    analyze_row_missingness,
    analyze_missing_patterns,
    analyze_missing_correlation,
)

from app.imputation.recommender import (
    recommend_imputation,
)

from app.imputation.executor import (
    execute_imputation,
)

from app.detection.rule_detector import (
    detect_rule_anomalies,
)

from app.detection.statistical_detector import (
    detect_iqr_anomalies,
)

from app.anomaly.hybrid_detector import (
    detect_hybrid_anomalies,
)

from app.cleaning.recommender import (
    recommend_cleaning,
)

from app.cleaning.executor import (
    execute_cleaning,
)

from app.quality.report import (
    build_quality_report,
)

from .models import PipelineResult


def run_pipeline(
    dataframe: pd.DataFrame,
    source_file: str = "memory",
    file_type: str = "dataframe",
) -> PipelineResult:

    # ==================================================
    # 0. VALIDATE INPUT
    # ==================================================

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    # ==================================================
    # 1. COPY ORIGINAL DATA
    # ==================================================

    original_dataframe = dataframe.copy()

    working_dataframe = dataframe.copy()

    # ==================================================
    # 2. CREATE UNIFIED TABLE
    # ==================================================

    table = UnifiedTable(
        dataframe=working_dataframe,
        source_file=source_file,
        file_type=file_type,
    )

    # ==================================================
    # 3. DATASET PROFILING
    # ==================================================

    dataset_profile = profile_dataset(
        table
    )

    profiles = (
        dataset_profile.column_profiles
    )

    # ==================================================
    # 4. SEMANTIC DETECTION
    # ==================================================

    profiles = detect_semantic_types(
        profiles
    )

    # Update profile object
    dataset_profile.column_profiles = profiles

    # ==================================================
    # 5. MISSING VALUE ANALYSIS
    # ==================================================

    missing_analysis = (
        analyze_missing_values(
            working_dataframe
        )
    )

    missing_row_analysis = (
        analyze_row_missingness(
            working_dataframe
        )
    )

    missing_patterns = (
        analyze_missing_patterns(
            working_dataframe
        )
    )

    missing_correlation = (
        analyze_missing_correlation(
            working_dataframe
        )
    )

    # ==================================================
    # 6. ANOMALY DETECTION
    # ==================================================

    rule_anomalies = detect_rule_anomalies(
        working_dataframe,
        profiles,
    )

    statistical_anomalies = (
        detect_iqr_anomalies(
            working_dataframe,
            profiles,
        )
    )

    anomalies = detect_hybrid_anomalies(
        rule_anomalies,
        statistical_anomalies,
    )

    # ==================================================
    # 7. CLEANING RECOMMENDATION
    # ==================================================

    cleaning_recommendations = []

    for anomaly in anomalies:

        anomaly_dict = {
            "row_index": anomaly.row_index,

            "column": anomaly.column,

            "value": anomaly.value,

            "anomaly_types": [
                anomaly.anomaly_type
            ],

            "severity": anomaly.severity,

            "score": anomaly.score,

            "reason": anomaly.reason,
        }

        recommendation = recommend_cleaning(
            anomaly_dict
        )

        cleaning_recommendations.append(
            recommendation
        )

    # ==================================================
    # 8. CLEANING EXECUTION
    # ==================================================

    cleaning_logs = []

    if cleaning_recommendations:

        working_dataframe, cleaning_logs = (
            execute_cleaning(
                working_dataframe,
                cleaning_recommendations,
            )
        )

    # ==================================================
    # 9. IMPUTATION RECOMMENDATION
    # ==================================================

    imputation_recommendations = (
        recommend_imputation(
            working_dataframe
        )
    )

    # ==================================================
    # 10. IMPUTATION EXECUTION
    # ==================================================

    imputation_logs = []

    if imputation_recommendations:

        (
            working_dataframe,
            imputation_logs,
        ) = execute_imputation(
            working_dataframe,
            recommendations=(
                imputation_recommendations
            ),
        )

    # ==================================================
    # 11. QUALITY REPORT
    # ==================================================

    quality_anomalies = [

        {
            "column": anomaly.column,

            "severity": anomaly.severity,

            "anomaly_types": [
                anomaly.anomaly_type
            ],
        }

        for anomaly in anomalies
    ]

    quality_report = build_quality_report(
        working_dataframe,
        quality_anomalies,
        cleaning_logs,
    )

    # ==================================================
    # 12. RETURN RESULT
    # ==================================================

    return PipelineResult(

        original_dataframe=(
            original_dataframe
        ),

        cleaned_dataframe=(
            working_dataframe
        ),

        dataset_profile=(
            dataset_profile
        ),

        missing_analysis=(
            missing_analysis
        ),

        missing_row_analysis=(
            missing_row_analysis
        ),

        missing_patterns=(
            missing_patterns
        ),

        missing_correlation=(
            missing_correlation
        ),

        rule_anomalies=(
            rule_anomalies
        ),

        statistical_anomalies=(
            statistical_anomalies
        ),

        anomalies=(
            anomalies
        ),

        cleaning_recommendations=(
            cleaning_recommendations
        ),

        cleaning_logs=(
            cleaning_logs
        ),

        imputation_recommendations=(
            imputation_recommendations
        ),

        imputation_logs=(
            imputation_logs
        ),

        quality_report=(
            quality_report
        ),
    )