import pandas as pd

from app.core.table import UnifiedTable
from app.pipeline.runner import run_pipeline


def print_separator(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():

    # ==========================================================
    # 1. CREATE SAMPLE DATASET
    # ==========================================================

    df = pd.DataFrame(
        {
            "customer_id": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],

            "age": [
                20,
                25,
                None,
                30,
                999,
                35,
            ],

            "salary": [
                10000000,
                12000000,
                None,
                15000000,
                100000000,
                None,
            ],

            "city": [
                "Da Nang",
                "Hue",
                None,
                "Da Nang",
                "Da Nang",
                None,
            ],
        }
    )

    # ==========================================================
    # 2. CREATE UNIFIED TABLE
    # ==========================================================

    table = UnifiedTable(
        dataframe=df,
        source_file="sample.csv",
        file_type="csv",
    )

    # ==========================================================
    # 3. ORIGINAL DATASET
    # ==========================================================

    print_separator("ORIGINAL DATASET")

    print(df)

    # ==========================================================
    # 4. RUN PIPELINE
    # ==========================================================

    result = run_pipeline(table)

    # ==========================================================
    # 5. DATASET PROFILE
    # ==========================================================

    print_separator("DATASET PROFILE")

    profile = result.dataset_profile

    print(f"Rows    : {profile.rows}")
    print(f"Columns : {profile.columns}")

    for column_profile in profile.column_profiles:

        print()
        print(
            f"Column: {column_profile.name}"
        )

        print(
            f"  dtype           : "
            f"{column_profile.dtype}"
        )

        print(
            f"  semantic_type   : "
            f"{column_profile.semantic_type}"
        )

        print(
            f"  missing_count   : "
            f"{column_profile.missing_count}"
        )

        print(
            f"  missing_ratio   : "
            f"{column_profile.missing_ratio:.2%}"
        )

        print(
            f"  unique_count    : "
            f"{column_profile.unique_count}"
        )

        print(
            f"  numeric_ratio   : "
            f"{column_profile.numeric_ratio:.2%}"
        )

        if column_profile.semantic_confidence:

            print(
                f"  semantic_conf.  : "
                f"{column_profile.semantic_confidence:.2f}"
            )

    # ==========================================================
    # 6. RULE ANOMALIES
    # ==========================================================

    print_separator("RULE ANOMALIES")

    if result.rule_anomalies:

        for anomaly in result.rule_anomalies:

            print(anomaly)

    else:

        print("No rule anomalies detected.")

    # ==========================================================
    # 7. STATISTICAL ANOMALIES
    # ==========================================================

    print_separator(
        "STATISTICAL ANOMALIES"
    )

    if result.statistical_anomalies:

        for anomaly in (
            result.statistical_anomalies
        ):

            print(anomaly)

    else:

        print(
            "No statistical anomalies detected."
        )

    # ==========================================================
    # 8. HYBRID ANOMALIES
    # ==========================================================

    print_separator(
        "HYBRID ANOMALIES"
    )

    if result.anomalies:

        for anomaly in result.anomalies:

            print(anomaly)

    else:

        print("No anomalies detected.")

    # ==========================================================
    # 9. CLEANING RECOMMENDATIONS
    # ==========================================================

    print_separator(
        "CLEANING RECOMMENDATIONS"
    )

    if result.cleaning_recommendations:

        for recommendation in (
            result.cleaning_recommendations
        ):

            print(recommendation)

    else:

        print(
            "No cleaning recommendations."
        )

    # ==========================================================
    # 10. CLEANING RESULTS
    # ==========================================================

    print_separator(
        "CLEANING RESULTS"
    )

    cleaning_results = getattr(
        result,
        "cleaning_results",
        getattr(
            result,
            "cleaning_logs",
            [],
        ),
    )

    if cleaning_results:

        for item in cleaning_results:

            print(item)

    else:

        print("No cleaning actions.")

    # ==========================================================
    # 11. IMPUTATION RECOMMENDATIONS
    # ==========================================================

    print_separator(
        "IMPUTATION RECOMMENDATIONS"
    )

    if result.imputation_recommendations:

        for recommendation in (
            result.imputation_recommendations
        ):

            print(recommendation)

    else:

        print(
            "No imputation recommendations."
        )

    # ==========================================================
    # 12. IMPUTATION RESULTS
    # ==========================================================

    print_separator(
        "IMPUTATION RESULTS"
    )

    imputation_results = getattr(
        result,
        "imputation_results",
        getattr(
            result,
            "imputation_logs",
            [],
        ),
    )

    if imputation_results:

        for item in imputation_results:

            print(item)

    else:

        print("No imputation actions.")

    # ==========================================================
    # 13. FINAL CLEANED DATASET
    # ==========================================================

    print_separator(
        "FINAL CLEANED DATASET"
    )

    final_df = getattr(
        result,
        "dataframe",
        result.cleaned_dataframe,
    )

    print(final_df)

    # ==========================================================
    # 14. QUALITY REPORT
    # ==========================================================

    print_separator(
        "QUALITY REPORT"
    )

    report = result.quality_report

    print(
        f"Rows              : "
        f"{report.rows}"
    )

    print(
        f"Columns           : "
        f"{report.columns}"
    )

    print(
        f"Total cells       : "
        f"{report.total_cells}"
    )

    print(
        f"Missing cells     : "
        f"{report.missing_cells}"
    )

    print(
        f"Missing ratio     : "
        f"{report.missing_ratio:.2%}"
    )

    print(
        f"Quality score     : "
        f"{report.quality_score:.2f}"
    )

    print(
        f"Total anomalies   : "
        f"{report.total_anomalies}"
    )

    print(
        f"Cleaned count     : "
        f"{report.cleaned_count}"
    )

    print(
        f"Review count      : "
        f"{report.review_count}"
    )

    print(
        f"Skipped count     : "
        f"{report.skipped_count}"
    )

    print()

    print("Severity counts:")

    print(
        report.severity_counts
    )

    print()

    print("Issue counts:")

    print(
        report.issue_counts
    )

    print()

    print("Column issues:")

    for column, issues in (
        report.column_issues.items()
    ):

        print(
            f"  {column}: {issues}"
        )

    # ==========================================================
    # 15. BEFORE / AFTER QUALITY EVALUATION
    # ==========================================================

    print_separator(
        "QUALITY EVALUATION"
    )

    comparison = (
        result.quality_comparison
    )

    if comparison is None:

        print(
            "Quality comparison is not available."
        )

    else:

        before = comparison.before
        after = comparison.after

        # ------------------------------------------------------
        # BEFORE
        # ------------------------------------------------------

        print("BEFORE")

        print(
            f"  Rows              : "
            f"{before.rows}"
        )

        print(
            f"  Columns           : "
            f"{before.columns}"
        )

        print(
            f"  Total cells       : "
            f"{before.total_cells}"
        )

        print(
            f"  Missing cells     : "
            f"{before.missing_cells}"
        )

        print(
            f"  Missing ratio     : "
            f"{before.missing_ratio:.2%}"
        )

        print(
            f"  Anomalies         : "
            f"{before.anomaly_count}"
        )

        print(
            f"  Quality score     : "
            f"{before.quality_score:.2f}"
        )

        # ------------------------------------------------------
        # AFTER
        # ------------------------------------------------------

        print()
        print("AFTER")

        print(
            f"  Rows              : "
            f"{after.rows}"
        )

        print(
            f"  Columns           : "
            f"{after.columns}"
        )

        print(
            f"  Total cells       : "
            f"{after.total_cells}"
        )

        print(
            f"  Missing cells     : "
            f"{after.missing_cells}"
        )

        print(
            f"  Missing ratio     : "
            f"{after.missing_ratio:.2%}"
        )

        print(
            f"  Anomalies         : "
            f"{after.anomaly_count}"
        )

        print(
            f"  Quality score     : "
            f"{after.quality_score:.2f}"
        )

        # ------------------------------------------------------
        # IMPROVEMENT
        # ------------------------------------------------------

        print()
        print("IMPROVEMENT")

        print(
            f"  Missing reduced       : "
            f"{comparison.missing_reduction}"
        )

        print(
            f"  Missing ratio reduced : "
            f"{comparison.missing_ratio_reduction:.2%}"
        )

        print(
            f"  Anomalies reduced     : "
            f"{comparison.anomaly_reduction}"
        )

        print(
            f"  Quality improvement   : "
            f"{comparison.quality_score_improvement:+.2f}"
        )

        print(
            f"  Quality improved      : "
            f"{comparison.quality_improved}"
        )

    # ==========================================================
    # 16. PIPELINE SUMMARY
    # ==========================================================

    print_separator(
        "PIPELINE SUMMARY"
    )

    print(
        f"Original rows       : "
        f"{len(df)}"
    )

    print(
        f"Final rows          : "
        f"{len(final_df)}"
    )

    print(
        f"Anomalies detected  : "
        f"{len(result.anomalies)}"
    )

    print(
        f"Cleaning actions    : "
        f"{len(cleaning_results)}"
    )

    print(
        f"Imputation actions  : "
        f"{len(imputation_results)}"
    )

    print(
        f"Quality score       : "
        f"{report.quality_score:.2f}"
    )

    if comparison is not None:

        print(
            f"Quality improvement: "
            f"{comparison.quality_score_improvement:+.2f}"
        )


if __name__ == "__main__":
    main()