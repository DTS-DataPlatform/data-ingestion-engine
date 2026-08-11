from collections import Counter

from .models import QualityReport


def build_quality_report(
    dataframe,
    anomalies: list,
    cleaning_results: list[dict],
) -> QualityReport:

    # ==========================================================
    # DATASET
    # ==========================================================

    rows = len(dataframe)

    columns = len(dataframe.columns)

    total_cells = rows * columns

    # ==========================================================
    # MISSING VALUES
    # ==========================================================

    missing_cells = int(
        dataframe.isna().sum().sum()
    )

    if total_cells > 0:

        missing_ratio = (
            missing_cells / total_cells
        )

    else:

        missing_ratio = 0.0

    # ==========================================================
    # ANOMALIES
    # ==========================================================

    total_anomalies = len(anomalies)

    # ==========================================================
    # QUALITY SCORE
    #
    # Quality consists of:
    #
    # 50% Missing-value quality
    # 50% Anomaly quality
    #
    # Missing quality:
    #
    #     1 - missing_ratio
    #
    # Anomaly quality:
    #
    #     1 - anomaly_ratio
    #
    # anomaly_ratio:
    #
    #     total_anomalies / total_cells
    #
    # ==========================================================

    # ----------------------------------------------------------
    # Missing quality
    # ----------------------------------------------------------

    if total_cells > 0:

        missing_quality = (
            1.0 - missing_ratio
        )

    else:

        missing_quality = 1.0

    # ----------------------------------------------------------
    # Anomaly quality
    # ----------------------------------------------------------

    if total_cells > 0:

        anomaly_ratio = (
            total_anomalies / total_cells
        )

        anomaly_quality = (
            1.0 - anomaly_ratio
        )

    else:

        anomaly_quality = 1.0

    # ----------------------------------------------------------
    # Final quality score
    # ----------------------------------------------------------

    quality_score = (
        0.5 * missing_quality
        +
        0.5 * anomaly_quality
    ) * 100

    # ----------------------------------------------------------
    # Keep score inside [0, 100]
    # ----------------------------------------------------------

    quality_score = max(
        0.0,
        min(100.0, quality_score)
    )

    # ==========================================================
    # COUNTERS
    # ==========================================================

    severity_counter = Counter()

    issue_counter = Counter()

    column_issues = {}

    # ==========================================================
    # ANALYZE ANOMALIES
    # ==========================================================

    for anomaly in anomalies:

        # ======================================================
        # SUPPORT DICT
        # ======================================================

        if isinstance(anomaly, dict):

            column = anomaly.get(
                "column"
            )

            severity = anomaly.get(
                "severity"
            )

            anomaly_types = anomaly.get(
                "anomaly_types",
                []
            )

            # --------------------------------------------------
            # Support singular anomaly_type
            # --------------------------------------------------

            if not anomaly_types:

                anomaly_type = anomaly.get(
                    "anomaly_type"
                )

                if anomaly_type:

                    anomaly_types = [
                        anomaly_type
                    ]

        # ======================================================
        # SUPPORT DATACLASS / OBJECT
        # ======================================================

        else:

            column = getattr(
                anomaly,
                "column",
                None
            )

            severity = getattr(
                anomaly,
                "severity",
                None
            )

            anomaly_types = getattr(
                anomaly,
                "anomaly_types",
                []
            )

            # --------------------------------------------------
            # Support AnomalyRecord
            # --------------------------------------------------

            if not anomaly_types:

                anomaly_type = getattr(
                    anomaly,
                    "anomaly_type",
                    None
                )

                if anomaly_type:

                    anomaly_types = [
                        anomaly_type
                    ]

        # ======================================================
        # SEVERITY COUNT
        # ======================================================

        if severity:

            severity_counter[
                severity
            ] += 1

        # ======================================================
        # ISSUE TYPE COUNT
        # ======================================================

        for anomaly_type in anomaly_types:

            issue_counter[
                anomaly_type
            ] += 1

        # ======================================================
        # IGNORE ANOMALIES WITHOUT COLUMN
        # ======================================================

        if column is None:

            continue

        # ======================================================
        # INITIALIZE COLUMN
        # ======================================================

        if column not in column_issues:

            column_issues[column] = {

                "anomalies": 0,

                "types": Counter(),

                "severities": Counter(),
            }

        # ======================================================
        # COLUMN ANOMALY COUNT
        # ======================================================

        column_issues[column][
            "anomalies"
        ] += 1

        # ======================================================
        # COLUMN ISSUE TYPES
        # ======================================================

        for anomaly_type in anomaly_types:

            column_issues[column][
                "types"
            ][anomaly_type] += 1

        # ======================================================
        # COLUMN SEVERITIES
        # ======================================================

        if severity:

            column_issues[column][
                "severities"
            ][severity] += 1

    # ==========================================================
    # CLEANING STATISTICS
    # ==========================================================

    cleaned_count = 0

    review_count = 0

    skipped_count = 0

    for result in cleaning_results:

        # ------------------------------------------------------
        # Safety
        # ------------------------------------------------------

        if not isinstance(result, dict):

            continue

        action = result.get(
            "action"
        )

        status = result.get(
            "status"
        )

        # ======================================================
        # CLEANED
        # ======================================================

        if (
            action == "replace_with_missing"
            or status == "cleaned"
        ):

            cleaned_count += 1

        # ======================================================
        # REVIEW
        # ======================================================

        elif (
            action == "review"
            or status == "review_required"
        ):

            review_count += 1

        # ======================================================
        # SKIPPED
        # ======================================================

        elif (
            action == "skip"
            or status == "skipped"
        ):

            skipped_count += 1

    # ==========================================================
    # NORMALIZE COLUMN ISSUES
    #
    # Counter is converted to normal dict so the result can be
    # serialized to JSON later by the web API.
    # ==========================================================

    normalized_column_issues = {}

    for column, data in column_issues.items():

        normalized_column_issues[column] = {

            "anomalies": data[
                "anomalies"
            ],

            "types": dict(
                data["types"]
            ),

            "severities": dict(
                data["severities"]
            ),
        }

    # ==========================================================
    # RETURN QUALITY REPORT
    # ==========================================================

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