from collections import defaultdict

from app.detection.agreement_models import (
    AnomalyAgreement,
)


def compare_detector_results(
    detector_results,
):
    """
    Compare anomaly results produced by
    multiple baseline detectors.

    Parameters
    ----------
    detector_results : dict[str, list[AnomalyRecord]]

        Example:

        {
            "iqr": [...],
            "zscore": [...],
            "lof": [...]
        }

    Returns
    -------
    list[AnomalyAgreement]
    """

    # ==========================================================
    # 1. COLLECT ALL DETECTOR ANOMALIES
    # ==========================================================

    groups = defaultdict(
        lambda: {
            "value": None,
            "detectors": [],
            "severity": "low",
        }
    )

    # ==========================================================
    # 2. GROUP BY ROW + COLUMN
    # ==========================================================

    for detector_name, anomalies in (
        detector_results.items()
    ):

        for anomaly in anomalies:

            key = (
                anomaly.row_index,
                anomaly.column,
            )

            groups[key]["value"] = (
                anomaly.value
            )

            groups[key][
                "detectors"
            ].append(
                detector_name
            )

            # --------------------------------------------------
            # Keep strongest severity
            # --------------------------------------------------

            severity_priority = {
                "low": 1,
                "medium": 2,
                "high": 3,
                "critical": 4,
            }

            current = groups[key][
                "severity"
            ]

            incoming = anomaly.severity

            if (
                severity_priority.get(
                    incoming,
                    0,
                )
                > severity_priority.get(
                    current,
                    0,
                )
            ):
                groups[key][
                    "severity"
                ] = incoming

    # ==========================================================
    # 3. NUMBER OF DETECTORS
    # ==========================================================

    total_detectors = len(
        detector_results
    )

    if total_detectors == 0:
        return []

    # ==========================================================
    # 4. BUILD AGREEMENT RESULTS
    # ==========================================================

    agreements = []

    for (
        row_index,
        column,
    ), data in groups.items():

        detectors = data[
            "detectors"
        ]

        agreement_count = len(
            detectors
        )

        agreement_ratio = (
            agreement_count
            / total_detectors
        )

        # ======================================================
        # CONFIDENCE
        # ======================================================

        confidence = agreement_ratio

        # ======================================================
        # BUILD RESULT
        # ======================================================

        agreements.append(
            AnomalyAgreement(
                row_index=row_index,

                column=column,

                value=data["value"],

                detectors=detectors,

                agreement_count=(
                    agreement_count
                ),

                agreement_ratio=(
                    agreement_ratio
                ),

                confidence=confidence,

                severity=data[
                    "severity"
                ],
            )
        )

    # ==========================================================
    # 5. SORT BY CONFIDENCE
    # ==========================================================

    agreements.sort(
        key=lambda item: (
            item.confidence,
            item.agreement_count,
        ),
        reverse=True,
    )

    return agreements