from collections import Counter, defaultdict

from .models import AnomalyRecord


def aggregate_anomalies(
    anomalies: list[AnomalyRecord],
) -> dict:
    """
    Aggregate and deduplicate anomaly records.

    Multiple detectors may detect the same
    row/column anomaly.

    Example:

        row 18, age
            rule
            statistical
            hybrid

    will be treated as ONE anomaly.
    """

    # ==========================================
    # 1. DEDUPLICATE
    # ==========================================

    grouped = {}

    for anomaly in anomalies:

        key = (
            anomaly.row_index,
            anomaly.column,
        )

        if key not in grouped:

            grouped[key] = {
                "row_index": anomaly.row_index,
                "column": anomaly.column,
                "value": anomaly.value,
                "anomaly_type": anomaly.anomaly_type,
                "score": anomaly.score,
                "severity": anomaly.severity,
                "detectors": [],
                "methods": [],
                "reasons": [],
            }

        item = grouped[key]

        # Detector
        if anomaly.detector not in item["detectors"]:
            item["detectors"].append(
                anomaly.detector
            )

        # Method
        if anomaly.method not in item["methods"]:
            item["methods"].append(
                anomaly.method
            )

        # Reason
        if anomaly.reason not in item["reasons"]:
            item["reasons"].append(
                anomaly.reason
            )

        # Keep highest score
        if anomaly.score > item["score"]:
            item["score"] = anomaly.score

        # Keep highest severity
        severity_rank = {
            "low": 1,
            "medium": 2,
            "high": 3,
        }

        current_rank = severity_rank.get(
            item["severity"],
            0,
        )

        new_rank = severity_rank.get(
            anomaly.severity,
            0,
        )

        if new_rank > current_rank:
            item["severity"] = anomaly.severity

    unique_anomalies = list(
        grouped.values()
    )

    # ==========================================
    # 2. EMPTY CASE
    # ==========================================

    if not unique_anomalies:

        return {
            "total_anomalies": 0,
            "severity": {},
            "detector": {},
            "anomaly_type": {},
            "columns": {},
            "records": [],
        }

    # ==========================================
    # 3. GLOBAL STATISTICS
    # ==========================================

    severity_counter = Counter(
        item["severity"]
        for item in unique_anomalies
    )

    detector_counter = Counter()

    anomaly_type_counter = Counter()

    for item in unique_anomalies:

        for detector in item["detectors"]:

            detector_counter[
                detector
            ] += 1

        anomaly_type_counter[
            item["anomaly_type"]
        ] += 1

    # ==========================================
    # 4. COLUMN STATISTICS
    # ==========================================

    column_data = defaultdict(list)

    for item in unique_anomalies:

        column_data[
            item["column"]
        ].append(item)

    columns = {}

    for column, records in column_data.items():

        severity = Counter(
            record["severity"]
            for record in records
        )

        detectors = Counter()

        anomaly_types = Counter()

        for record in records:

            for detector in record[
                "detectors"
            ]:

                detectors[detector] += 1

            anomaly_types[
                record["anomaly_type"]
            ] += 1

        columns[column] = {

            "total": len(records),

            "severity": dict(
                severity
            ),

            "detectors": dict(
                detectors
            ),

            "anomaly_types": dict(
                anomaly_types
            ),

            "rows": [
                record["row_index"]
                for record in records
            ],
        }

    # ==========================================
    # 5. FINAL RESULT
    # ==========================================

    return {

        "total_anomalies": len(
            unique_anomalies
        ),

        "severity": dict(
            severity_counter
        ),

        "detector": dict(
            detector_counter
        ),

        "anomaly_type": dict(
            anomaly_type_counter
        ),

        "columns": columns,

        "records": unique_anomalies,
    }
