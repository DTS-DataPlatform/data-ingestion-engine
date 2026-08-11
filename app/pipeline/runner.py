from app.profiling.dataset_profiler import (
    profile_dataset,
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

from app.imputation.recommender import (
    recommend_imputation,
)

from app.imputation.executor import (
    execute_imputation,
)

from app.quality.report import (
    build_quality_report,
)

from app.quality.comparison import (
    create_quality_snapshot,
    compare_quality,
)

from .models import PipelineResult


def run_pipeline(table) -> PipelineResult:

    # ==================================================
    # 1. PROFILE DATASET
    # ==================================================

    dataset_profile = profile_dataset(
        table
    )

    profiles = (
        dataset_profile.column_profiles
    )

    # ==================================================
    # 2. RULE ANOMALY DETECTION
    # ==================================================

    rule_anomalies = detect_rule_anomalies(
        table.dataframe,
        profiles,
    )

    # ==================================================
    # 3. STATISTICAL ANOMALY DETECTION
    # ==================================================

    statistical_anomalies = detect_iqr_anomalies(
        table.dataframe,
        profiles,
    )

    # ==================================================
    # 4. HYBRID ANOMALY DETECTION
    # ==================================================

    anomalies = detect_hybrid_anomalies(
        rule_anomalies,
        statistical_anomalies,
    )

    before_quality = create_quality_snapshot(
        dataframe=table.dataframe,
        anomaly_count=len(anomalies),
    )
    # ==================================================
    # 5. CLEANING RECOMMENDATIONS
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
            "score": anomaly.score,
            "severity": anomaly.severity,
            "reason": anomaly.reason,
        }

        recommendation = (
            recommend_cleaning(
                anomaly_dict
            )
        )

        cleaning_recommendations.append(
            recommendation
        )

    # ==================================================
    # 6. EXECUTE CLEANING
    # ==================================================

    cleaned_df, cleaning_results = (
        execute_cleaning(
            table.dataframe,
            cleaning_recommendations,
        )
    )

    # ==================================================
    # 7. IMPUTATION RECOMMENDATIONS
    # ==================================================

    imputation_recommendations = (
        recommend_imputation(
            cleaned_df
        )
    )

    # ==================================================
    # 8. EXECUTE IMPUTATION
    # ==================================================

    imputed_df, imputation_results = (
        execute_imputation(
            cleaned_df,
            recommendations=(
                imputation_recommendations
            ),
        )
    )
    
    after_quality = create_quality_snapshot(
        dataframe=imputed_df,
        anomaly_count=0,
    )
    
    quality_comparison = compare_quality(
        before=before_quality,
        after=after_quality,
    )

    # ==================================================
    # 9. QUALITY REPORT
    # ==================================================

    quality_report = build_quality_report(
        dataframe=imputed_df,
        anomalies=anomalies,
        cleaning_results=cleaning_results,
    )

    # ==================================================
    # 10. RETURN PIPELINE RESULT
    # ==================================================
    return PipelineResult(
        original_dataframe=table.dataframe,

        cleaned_dataframe=imputed_df,

        dataset_profile=dataset_profile,

        rule_anomalies=rule_anomalies,

        statistical_anomalies=(
            statistical_anomalies
        ),

        anomalies=anomalies,

        cleaning_recommendations=(
            cleaning_recommendations
        ),

        cleaning_logs=cleaning_results,

        imputation_recommendations=(
            imputation_recommendations
        ),

        imputation_logs=imputation_results,

        quality_report=quality_report,

        quality_comparison=quality_comparison,
    )