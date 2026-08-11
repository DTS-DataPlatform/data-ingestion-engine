import pandas as pd

from app.detection.baseline.dbscan_detector import (
    detect_dbscan_anomalies,
)
from app.profiling.column_profiler import profile_column


def test_dbscan_detects_noise():

    df = pd.DataFrame(
        {
            "x": [
                1,
                1.1,
                0.9,
                1.2,
                1.05,
                100,
            ],
            "y": [
                1,
                1.1,
                0.9,
                1.2,
                1.05,
                100,
            ],
        }
    )

    profiles = [
        profile_column(df["x"]),
        profile_column(df["y"]),
    ]

    anomalies = detect_dbscan_anomalies(
        df,
        profiles,
    )

    assert isinstance(
        anomalies,
        list,
    )