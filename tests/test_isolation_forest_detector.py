import pandas as pd

from app.detection.baseline.isolation_forest_detector import (
    detect_isolation_forest_anomalies,
)
from app.profiling.column_profiler import profile_column


def test_isolation_forest_detects_outlier():

    df = pd.DataFrame(
        {
            "age": [
                20,
                21,
                22,
                23,
                24,
                25,
                999,
            ],
            "salary": [
                10,
                11,
                10,
                12,
                11,
                10,
                1000,
            ],
        }
    )

    profiles = [
        profile_column(df["age"]),
        profile_column(df["salary"]),
    ]

    anomalies = detect_isolation_forest_anomalies(
        df,
        profiles,
    )

    assert isinstance(
        anomalies,
        list,
    )

    assert len(anomalies) > 0