from app.core.table import UnifiedTable
from app.profiling.dataset_profiler import profile_dataset
from app.detection.rule_detector import (
    detect_rule_anomalies
)
from app.detection.statistical_detector import (
    detect_iqr_anomalies
)
from app.anomaly.hybrid_detector import (
    detect_hybrid_anomalies
)
from app.anomaly.aggregator import aggregate_anomalies
from app.anomaly.deduplicator import ( deduplicate_anomalies )
from app.cleaning.recommender import (
    recommend_cleaning
)
from app.cleaning.executor import execute_cleaning
from app.quality.report import (
    build_quality_report
)

import pandas as pd


df = pd.read_csv(
    "storage/data/data.csv"
)
df = df.sample(n=1000, random_state=42).reset_index(drop=True)

table = UnifiedTable(
    dataframe=df,
    source_file="storage/data/data.csv",
    file_type="csv"
)


dataset_profile = profile_dataset(
    table
)

anomalies = detect_rule_anomalies(
    table.dataframe,
    dataset_profile.column_profiles
)

statistical_anomalies = detect_iqr_anomalies(
    table.dataframe,
    dataset_profile.column_profiles
)

hybrid_anomalies = detect_hybrid_anomalies(
    anomalies,
    statistical_anomalies
)

# ==========================================
# COMBINE ALL DETECTIONS
# ==========================================

all_anomalies = (
    anomalies
    + statistical_anomalies
    + hybrid_anomalies
)

# ==========================================
# DEDUPLICATION
# ==========================================

final_anomalies = deduplicate_anomalies(
    all_anomalies
)

# ==========================================
# ANOMALY DEDUPLICATION
# ==========================================



# print("\n")
# print("=" * 60)
# print("FINAL ANOMALIES")
# print("=" * 60)

# print(
#     f"Total final anomalies: "
#     f"{len(final_anomalies)}"
# )


# for anomaly in final_anomalies:

#     print("\n")

#     print(
#         f"Row: "
#         f"{anomaly['row_index']}"
#     )

#     print(
#         f"Column: "
#         f"{anomaly['column']}"
#     )

#     print(
#         f"Value: "
#         f"{anomaly['value']}"
#     )

#     print(
#         f"Type: "
#         f"{anomaly['anomaly_types']}"
#     )

#     print(
#         f"Detectors: "
#         f"{anomaly['detectors']}"
#     )

#     print(
#         f"Methods: "
#         f"{anomaly['methods']}"
#     )

#     print(
#         f"Score: "
#         f"{anomaly['score']:.3f}"
#     )

#     print(
#         f"Severity: "
#         f"{anomaly['severity']}"
#     )

#     print(
#         f"Detection count: "
#         f"{anomaly['detection_count']}"
#     )

#     print(
#         "Reasons:"
#     )

#     for reason in anomaly["reasons"]:

#         print(
#             f"  - {reason}"
#         )

recommendations = []

for anomaly in final_anomalies:

    recommendation = recommend_cleaning(
        anomaly
    )

    recommendations.append(
        recommendation
    )
cleaned_df, cleaning_log = execute_cleaning(
    table.dataframe,
    recommendations,
)

quality_report = build_quality_report(
    dataframe=cleaned_df,
    anomalies=final_anomalies,
    cleaning_results=cleaning_log,
)
print("\n")
print("=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)

print(
    f"Rows: "
    f"{quality_report.rows}"
)

print(
    f"Columns: "
    f"{quality_report.columns}"
)

print(
    f"Total cells: "
    f"{quality_report.total_cells}"
)

print(
    f"Missing cells: "
    f"{quality_report.missing_cells}"
)

print(
    f"Missing ratio: "
    f"{quality_report.missing_ratio:.3f}"
)

print(
    f"Quality score: "
    f"{quality_report.quality_score:.2f}"
)

print("\nANOMALIES")

print(
    f"Total: "
    f"{quality_report.total_anomalies}"
)

print(
    f"Severity: "
    f"{quality_report.severity_counts}"
)

print(
    f"Issue types: "
    f"{quality_report.issue_counts}"
)

print("\nCLEANING")

print(
    f"Cleaned: "
    f"{quality_report.cleaned_count}"
)

print(
    f"Requires review: "
    f"{quality_report.review_count}"
)

print(
    f"Skipped: "
    f"{quality_report.skipped_count}"
)

print("\nCOLUMN ISSUES")

for column, info in (
    quality_report.column_issues.items()
):

    print(
        f"\n{column}"
    )

    print(
        f"  Anomalies: "
        f"{info['anomalies']}"
    )

    print(
        f"  Types: "
        f"{info['types']}"
    )

    print(
        f"  Severities: "
        f"{info['severities']}"
    )