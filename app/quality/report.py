from collections import Counter

from .models import QualityReport


def build_quality_report(
    dataframe,
    anomalies: list[dict],
    cleaning_results: list[dict],
) -> QualityReport:

    # ==========================================
    # DATASET
    # ==========================================

    rows = len(dataframe)

    columns = len(dataframe.columns)

    total_cells = rows * columns

    # ==========================================
    # MISSING
    # ==========================================

    missing_cells = int(
        dataframe.isna().sum().sum()
    )

    if total_cells > 0:

        missing_ratio = (
            missing_cells / total_cells
        )

    else:

        missing_ratio = 0.0

    # ==========================================
    # QUALITY SCORE
    # ==========================================

    if total_cells > 0:

        quality_score = (
            (total_cells - missing_cells)
            / total_cells
        ) * 100

    else:

        quality_score = 100.0

    # ==========================================
    # ANOMALIES
    # ==========================================

    total_anomalies = len(anomalies)

    # ==========================================
    # SEVERITY
    # ==========================================

    severity_counter = Counter()

    # ==========================================
    # ISSUE TYPES
    # ==========================================

    issue_counter = Counter()

    # ==========================================
    # COLUMN ISSUES
    # ==========================================

    column_issues = {}

    for anomaly in anomalies:

        column = anomaly.get(
            "column"
        )

        severity = anomaly.get(
            "severity"
        )

        if severity:

            severity_counter[
                severity
            ] += 1

        anomaly_types = anomaly.get(
            "anomaly_types",
            []
        )

        for anomaly_type in anomaly_types:

            issue_counter[
                anomaly_type
            ] += 1

        # --------------------------------------
        # Column
        # --------------------------------------

        if column not in column_issues:

            column_issues[column] = {
                "anomalies": 0,
                "types": Counter(),
                "severities": Counter(),
            }

        column_issues[column][
            "anomalies"
        ] += 1

        for anomaly_type in anomaly_types:

            column_issues[column][
                "types"
            ][anomaly_type] += 1

        if severity:

            column_issues[column][
                "severities"
            ][severity] += 1

    # ==========================================
    # CLEANING
    # ==========================================

    cleaned_count = 0

    review_count = 0

    skipped_count = 0

    for result in cleaning_results:

        action = result.get(
            "action"
        )

        if action == "replace_with_missing":

            cleaned_count += 1

        elif action == "review":

            review_count += 1

        elif action == "skip":

            skipped_count += 1

    # ==========================================
    # NORMALIZE COLUMN COUNTERS
    # ==========================================

    normalized_column_issues = {}

    for column, data in column_issues.items():

        normalized_column_issues[column] = {

            "anomalies": data["anomalies"],

            "types": dict(
                data["types"]
            ),

            "severities": dict(
                data["severities"]
            ),
        }

    # ==========================================
    # RETURN REPORT
    # ==========================================

    return QualityReport(

        rows=rows,

        columns=columns,

        total_cells=total_cells,

        missing_cells=missing_cells,

        missing_ratio=missing_ratio,

        quality_score=quality_score,

        total_anomalies=total_anomalies,

        cleaned_count=cleaned_count,

        review_count=review_count,

        skipped_count=skipped_count,

        severity_counts=dict(
            severity_counter
        ),

        issue_counts=dict(
            issue_counter
        ),

        column_issues=normalized_column_issues,
    )