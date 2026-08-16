def classify_anomaly(anomaly: dict) -> str:

    anomaly_types = set(
        anomaly.get("anomaly_types", [])
    )

    if "invalid_value" in anomaly_types:
        return "invalid"

    if "outlier" in anomaly_types:
        return "outlier"

    return "unknown"