import pandas as pd

from types import SimpleNamespace

from app.profiling.column_profiler import profile_column

from app.detection.baseline_runner import (
    run_baseline_detection,
)

from app.detection.hybrid.aggregator import (
    aggregate_anomalies,
)

from app.detection.hybrid.score_normalizer import (
    normalize_detector_score,
)


# ==========================================================
# PRINT HELPERS
# ==========================================================

def print_separator(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_anomalies(anomalies):
    if not anomalies:
        print("No anomalies detected.")
        return

    for anomaly in anomalies:
        print(
            f"row={anomaly.row_index}, "
            f"column={anomaly.column}, "
            f"value={anomaly.value}, "
            f"score={anomaly.score:.4f}, "
            f"severity={anomaly.severity}"
        )

        print(
            f"  method   : {anomaly.method}"
        )

        print(
            f"  detector : {anomaly.detector}"
        )

        print(
            f"  reason   : {anomaly.reason}"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # 1. CREATE DATASET
    # ======================================================

    df = pd.DataFrame(
        {
            "customer_id": list(range(1, 51)),

            "age": [
                20, 21, 22, 23, 24,
                25, 26, 27, 28, 29,
                30, 31, 32, 33, 34,
                35, 36, 37, 38, 39,
                40, 41, 42, 43, 44,
                45, 46, 47, 48, 49,
                50, 51, 52, 53, 54,
                55, 56, 57, 58, 59,
                60, 61, 62, 63, 64,
                65, 66, 67, 68, 999,
            ],

            "salary": [
                10_000_000,
                10_500_000,
                11_000_000,
                11_500_000,
                12_000_000,
                12_500_000,
                13_000_000,
                13_500_000,
                14_000_000,
                14_500_000,
                15_000_000,
                15_500_000,
                16_000_000,
                16_500_000,
                17_000_000,
                17_500_000,
                18_000_000,
                18_500_000,
                19_000_000,
                19_500_000,
                20_000_000,
                20_500_000,
                21_000_000,
                21_500_000,
                22_000_000,
                22_500_000,
                23_000_000,
                23_500_000,
                24_000_000,
                24_500_000,
                25_000_000,
                25_500_000,
                26_000_000,
                26_500_000,
                27_000_000,
                27_500_000,
                28_000_000,
                28_500_000,
                29_000_000,
                29_500_000,
                30_000_000,
                30_500_000,
                31_000_000,
                31_500_000,
                32_000_000,
                32_500_000,
                33_000_000,
                33_500_000,
                34_000_000,
                200_000_000,
            ],

            "experience": [
                1, 2, 1, 3, 2,
                4, 3, 5, 4, 6,
                5, 7, 6, 8, 7,
                9, 8, 10, 9, 11,
                10, 12, 11, 13, 12,
                14, 13, 15, 14, 16,
                15, 17, 16, 18, 17,
                19, 18, 20, 19, 21,
                20, 22, 21, 23, 22,
                24, 23, 25, 24, 30,
            ],
        }
    )

    # ======================================================
    # 2. ORIGINAL DATASET
    # ======================================================

    print_separator("ORIGINAL DATASET")

    print(df)

    print()
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # ======================================================
    # 3. PROFILE
    # ======================================================

    print_separator("DATASET PROFILING")

    profiles = []

    for column in df.columns:

        profile = profile_column(
            df[column]
        )

        profiles.append(profile)

        print(
            f"\nColumn: {column}"
        )

        print(
            f"  dtype         : {profile.dtype}"
        )

        print(
            f"  mean          : {profile.mean}"
        )

        print(
            f"  missing_count : {profile.missing_count}"
        )

        print(
            f"  unique_count  : {profile.unique_count}"
        )

    # ======================================================
    # 4. DATASET CHARACTERISTICS
    # ======================================================

    numeric_columns = 0

    for profile in profiles:

        if profile.mean is not None:
            numeric_columns += 1

    characteristics = SimpleNamespace(
        rows=len(df),
        numeric_columns=numeric_columns,
        skewed=False,
    )

    print_separator(
        "DATASET CHARACTERISTICS"
    )

    print(
        f"Rows             : "
        f"{characteristics.rows}"
    )

    print(
        f"Numeric columns  : "
        f"{characteristics.numeric_columns}"
    )

    print(
        f"Skewed           : "
        f"{characteristics.skewed}"
    )

    # ======================================================
    # 5. BASELINE DETECTION
    # ======================================================

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    selected_detectors = results[
        "selected_detectors"
    ]

    print_separator(
        "SELECTED DETECTORS"
    )

    for detector in selected_detectors:

        print(
            f"- {detector}"
        )

    # ======================================================
    # 6. PRINT EACH DETECTOR
    # ======================================================

    print_separator(
        "BASELINE ANOMALY DETECTION"
    )

    raw_anomalies = []

    for detector in selected_detectors:

        anomalies = results.get(
            detector,
            []
        )

        print()
        print(
            f"--- {detector.upper()} ---"
        )

        print(
            f"Detected: {len(anomalies)}"
        )

        print()

        print_anomalies(
            anomalies
        )

        raw_anomalies.extend(
            anomalies
        )

    # ======================================================
    # 7. NORMALIZED SCORES
    # ======================================================

    print_separator(
        "NORMALIZED DETECTOR SCORES"
    )

    for detector in selected_detectors:

        anomalies = results.get(
            detector,
            []
        )

        if not anomalies:
            continue

        print()
        print(
            f"--- {detector.upper()} ---"
        )

        for anomaly in anomalies:

            # ------------------------------------------------
            # DBSCAN does not use the same score scale
            # as IQR / Z-score / LOF / Isolation Forest.
            #
            # Therefore skip it until a dedicated DBSCAN
            # normalization function is implemented.
            # ------------------------------------------------

            if detector == "dbscan":

                print(
                    f"row={anomaly.row_index}, "
                    f"raw_score={anomaly.score}, "
                    f"normalized=N/A"
                )

                continue

            normalized = (
                normalize_detector_score(
                    detector,
                    anomaly.score,
                )
            )

            print(
                f"row={anomaly.row_index}, "
                f"column={anomaly.column}, "
                f"raw={anomaly.score:.4f}, "
                f"normalized={normalized:.4f}"
            )

    # ======================================================
    # 8. HYBRID AGGREGATION
    # ======================================================

    print_separator(
        "HYBRID ANOMALY AGGREGATION"
    )

    hybrid_results = aggregate_anomalies(
        raw_anomalies,
        selected_detectors,
    )

    if not hybrid_results:

        print(
            "No hybrid anomalies detected."
        )

    else:

        for result in hybrid_results:

            print()

            print(
                f"Row              : "
                f"{result.row_index}"
            )

            print(
                f"Column           : "
                f"{result.column}"
            )

            print(
                f"Value            : "
                f"{result.value}"
            )

            print(
                f"Detectors        : "
                f"{result.detectors}"
            )

            print(
                f"Detector count   : "
                f"{result.detector_count}"
            )

            print(
                f"Total detectors  : "
                f"{result.total_detectors}"
            )

            print(
                f"Agreement ratio  : "
                f"{result.agreement_ratio:.4f}"
            )

            print(
                f"Confidence       : "
                f"{result.confidence:.4f}"
            )

            print(
                f"Anomaly type     : "
                f"{result.anomaly_type}"
            )

            print(
                f"Severity         : "
                f"{result.severity}"
            )

            print(
                f"Reason           : "
                f"{result.reason}"
            )

    # ======================================================
    # 9. SUMMARY
    # ======================================================

    print_separator(
        "PIPELINE SUMMARY"
    )

    print(
        f"Rows               : {len(df)}"
    )

    print(
        f"Columns            : {len(df.columns)}"
    )

    print(
        f"Selected detectors : "
        f"{len(selected_detectors)}"
    )

    print(
        f"Raw anomalies      : "
        f"{len(raw_anomalies)}"
    )

    print(
        f"Hybrid anomalies   : "
        f"{len(hybrid_results)}"
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()