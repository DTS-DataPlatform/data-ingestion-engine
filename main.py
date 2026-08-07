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

print(f"RULE: {len(anomalies)}")
print(f"STATISTICAL: {len(statistical_anomalies)}")
print(f"HYBRID: {len(hybrid_anomalies)}")
print(f"FINAL: {len(final_anomalies)}")