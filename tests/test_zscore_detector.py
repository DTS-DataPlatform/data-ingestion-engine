import pandas as pd

from app.detection.baseline.zscore_detector import (
    detect_zscore_anomalies,
)
from app.profiling.models import ColumnProfile


def make_profile(name):

    return ColumnProfile(
        name=name,
        dtype="float64",
        semantic_type=None,
        missing_count=0,
        missing_ratio=0.0,
        unique_count=6,
        unique_ratio=1.0,
        numeric_ratio=1.0,
        mean=0.0,
    )

def test_zscore_detects_outlier():

    df = pd.DataFrame(
        {
            "age": [
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                100,
            ]
        }
    )

    profile = make_profile("age")

    anomalies = detect_zscore_anomalies(
        df,
        [profile],
    )

    assert len(anomalies) >= 1
    assert anomalies[0].column == "age"
    assert anomalies[0].value == 100
    assert anomalies[0].method == "Z-score" 
    
    
    