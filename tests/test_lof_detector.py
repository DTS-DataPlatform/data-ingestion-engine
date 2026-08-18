import pandas as pd

from app.detection.baseline.lof_detector import (
    detect_lof_anomalies,
)
from app.profiling.column_profiler import profile_column


def test_lof_detects_outlier():

    df = pd.DataFrame(
        {
            "x": [
                1,
                2,
                1,
                2,
                1,
                2,
                100,
            ],
            "y": [
                1,
                2,
                2,
                1,
                1,
                2,
                100,
            ],
        }
    )

    profiles = [
        profile_column(df["x"]),
        profile_column(df["y"]),
    ]

    anomalies = detect_lof_anomalies(
        df,
        profiles,
    )

    assert isinstance(
        anomalies,
        list,
    )

    assert len(anomalies) > 0