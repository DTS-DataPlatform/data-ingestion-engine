import pandas as pd

from .models import AnomalyRecord
from .rule_registry import RULES
from app.anomaly.scoring import (
    calculate_rule_score,
    calculate_severity,
)


def detect_rule_anomalies(
    df: pd.DataFrame,
    profiles
):

    anomalies = []

    for profile in profiles:

        semantic_type = profile.semantic_type

        # ==========================================
        # No rule
        # ==========================================

        if semantic_type not in RULES:
            continue

        rules = RULES[semantic_type]

        series = pd.to_numeric(
            df[profile.name],
            errors="coerce"
        )

        for index, value in series.items():

            # Ignore missing / invalid conversion
            if pd.isna(value):
                continue

            # ======================================
            # MIN RULE
            # ======================================

            if (
                "min" in rules
                and value < rules["min"]
            ):

                score = calculate_rule_score(
                    value=value,
                    rules=rules,
                )

                severity = calculate_severity(
                    score
                )

                anomalies.append(
                    AnomalyRecord(
                        row_index=int(index),

                        column=profile.name,

                        value=float(value),

                        anomaly_type="invalid_value",

                        detector="rule",

                        score=float(score),

                        method="min_rule",

                        severity=severity,

                        reason=(
                            f"{semantic_type} "
                            f"must be >= "
                            f"{rules['min']}"
                        ),
                    )
                )

                continue

            # ======================================
            # MAX RULE
            # ======================================

            if (
                "max" in rules
                and value > rules["max"]
            ):

                score = calculate_rule_score(
                    value=value,
                    rules=rules,
                )

                severity = calculate_severity(
                    score
                )

                anomalies.append(
                    AnomalyRecord(
                        row_index=int(index),

                        column=profile.name,

                        value=float(value),

                        anomaly_type="invalid_value",

                        detector="rule",

                        score=float(score),

                        method="max_rule",

                        severity=severity,

                        reason=(
                            f"{semantic_type} "
                            f"must be <= "
                            f"{rules['max']}"
                        ),
                    )
                )

    return anomalies