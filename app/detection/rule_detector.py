import pandas as pd

from .models import AnomalyRecord
from .rule_registry import RULES


def detect_rule_anomalies(
    df: pd.DataFrame,
    profiles
):

    anomalies = []

    for profile in profiles:

        semantic_type = profile.semantic_type

        # ==========================================
        # Không có semantic type
        # ==========================================

        if semantic_type is None:
            continue

        # ==========================================
        # Không có rule tương ứng
        # ==========================================

        if semantic_type not in RULES:
            continue

        rules = RULES[semantic_type]

        # ==========================================
        # Convert sang numeric
        # ==========================================

        series = pd.to_numeric(
            df[profile.name],
            errors="coerce"
        )

        # ==========================================
        # Duyệt từng giá trị
        # ==========================================

        for index, value in series.items():

            # Bỏ qua missing / không convert được
            if pd.isna(value):
                continue

            # ======================================
            # MIN RULE
            # ======================================

            if (
                "min" in rules
                and value < rules["min"]
            ):

                anomalies.append(
                    AnomalyRecord(
                        row_index=int(index),
                        column=profile.name,
                        value=value,

                        anomaly_type="invalid_value",

                        detector="rule",

                        score=1.0,

                        method="min_rule",

                        severity="high",

                        reason=(
                            f"{semantic_type} "
                            f"must be >= {rules['min']}"
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

                anomalies.append(
                    AnomalyRecord(
                        row_index=int(index),
                        column=profile.name,
                        value=value,

                        anomaly_type="invalid_value",

                        detector="rule",

                        score=1.0,

                        method="max_rule",

                        severity="high",

                        reason=(
                            f"{semantic_type} "
                            f"must be <= {rules['max']}"
                        ),
                    )
                )

    return anomalies