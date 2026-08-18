from app.core.table import UnifiedTable
from app.profiling.dataset_profiler import profile_dataset
from app.detection.rule_detector import (
    detect_rule_anomalies
)
from app.detection.statistical_detector import (
    detect_iqr_anomalies
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

print("Rows:", dataset_profile.rows)
print("Columns:", dataset_profile.columns)


# print("\nCOLUMN PROFILES:")

# for profile in dataset_profile.column_profiles:

#     print(profile)


# print("\nCORRELATION:")

# print(
#     dataset_profile.correlation
# )

# print("\nRULE ANOMALIES:")

# for anomaly in anomalies:

#     print(anomaly)
    
print("\nSTATISTICAL ANOMALIES:")

for anomaly in statistical_anomalies:

    print(anomaly)