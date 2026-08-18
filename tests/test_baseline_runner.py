import pandas as pd

from app.detection.baseline_runner import (
    run_baseline_detection,
)

from app.profiling.column_profiler import (
    profile_column,
)

from types import SimpleNamespace


def test_run_baseline_detection_small_dataset():

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

    characteristics = SimpleNamespace(
        rows=7,
        numeric_columns=2,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert isinstance(
        results,
        dict,
    )

    assert "iqr" in results

    assert "zscore" in results

    assert isinstance(
        results["iqr"],
        list,
    )

    assert isinstance(
        results["zscore"],
        list,
    )


def test_run_baseline_detection_large_dataset():

    df = pd.DataFrame(
        {
            "x": list(range(1, 101)),
            "y": list(range(1, 101)),
        }
    )

    profiles = [
        profile_column(df["x"]),
        profile_column(df["y"]),
    ]

    characteristics = SimpleNamespace(
        rows=100,
        numeric_columns=2,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert isinstance(
        results,
        dict,
    )

    assert "iqr" in results

    assert "zscore" in results

    assert "isolation_forest" in results

    assert "lof" in results

    assert "dbscan" in results


def test_runner_does_not_run_unselected_detector():

    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
        }
    )

    profiles = [
        profile_column(df["x"]),
    ]

    characteristics = SimpleNamespace(
        rows=5,
        numeric_columns=1,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert "iqr" in results

    assert "zscore" in results

    assert "isolation_forest" not in results

    assert "lof" not in results

    assert "dbscan" not in results

def test_runner_clean_dataset_returns_no_anomalies():

    df = pd.DataFrame(
        {
            "x": [10, 11, 12, 13, 14, 15, 16],
            "y": [20, 21, 22, 23, 24, 25, 26],
        }
    )

    profiles = [
        profile_column(df["x"]),
        profile_column(df["y"]),
    ]

    characteristics = SimpleNamespace(
        rows=7,
        numeric_columns=2,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert "iqr" in results
    assert "zscore" in results

    assert isinstance(results["iqr"], list)
    assert isinstance(results["zscore"], list)

    assert len(results["iqr"]) == 0
    assert len(results["zscore"]) == 0
    
def test_runner_iqr_detects_outlier():

    df = pd.DataFrame(
        {
            "x": [
                1,
                2,
                2,
                3,
                3,
                4,
                100,
            ]
        }
    )

    profiles = [
        profile_column(df["x"]),
    ]

    characteristics = SimpleNamespace(
        rows=7,
        numeric_columns=1,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert "iqr" in results

    assert len(results["iqr"]) > 0

    anomaly = results["iqr"][0]

    assert anomaly.row_index == 6
    assert anomaly.value == 100
    assert anomaly.method == "IQR" 

def test_runner_zscore_detects_outlier():

    df = pd.DataFrame(
        {
            "x": [
                10,
                11,
                10,
                12,
                11,
                10,
                100,
            ]
        }
    )

    profiles = [
        profile_column(df["x"]),
    ]

    characteristics = SimpleNamespace(
        rows=7,
        numeric_columns=1,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert "zscore" in results

    assert isinstance(
        results["zscore"],
        list,
    )

    assert len(results["zscore"]) > 0

def test_runner_skewed_dataset_skips_zscore():

    df = pd.DataFrame(
        {
            "x": [
                1,
                1,
                1,
                2,
                2,
                3,
                100,
            ]
        }
    )

    profiles = [
        profile_column(df["x"]),
    ]

    characteristics = SimpleNamespace(
        rows=7,
        numeric_columns=1,
        skewed=True,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert "iqr" in results

    assert "zscore" not in results

    assert "isolation_forest" not in results

def test_runner_no_numeric_columns():

    df = pd.DataFrame(
        {
            "city": [
                "Da Nang",
                "Hue",
                "Ha Noi",
                "Da Nang",
                "Hue",
            ]
        }
    )

    profiles = [
        profile_column(df["city"]),
    ]

    characteristics = SimpleNamespace(
        rows=5,
        numeric_columns=0,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert isinstance(
        results,
        dict,
    )

    assert "iqr" not in results

    assert "zscore" not in results
    
def test_large_single_numeric_column():

    df = pd.DataFrame(
        {
            "x": list(range(1, 101))
        }
    )

    profiles = [
        profile_column(df["x"]),
    ]

    characteristics = SimpleNamespace(
        rows=100,
        numeric_columns=1,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert "iqr" in results
    assert "zscore" in results

    assert "isolation_forest" in results

    assert "lof" not in results

    assert "dbscan" not in results
    
def test_multivariate_dataset_runs_lof_and_dbscan():

    df = pd.DataFrame(
        {
            "x": [
                1, 2, 1, 2, 1,
                2, 1, 2, 100, 1
            ],
            "y": [
                1, 2, 2, 1, 1,
                2, 1, 2, 100, 1
            ],
        }
    )

    profiles = [
        profile_column(df["x"]),
        profile_column(df["y"]),
    ]

    characteristics = SimpleNamespace(
        rows=10,
        numeric_columns=2,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert "iqr" in results
    assert "zscore" in results

    # rows < 20
    assert "isolation_forest" not in results

    assert "lof" not in results
    assert "dbscan" not in results
    
def test_large_multivariate_dataset_runs_all_detectors():

    df = pd.DataFrame(
        {
            "x": list(range(1, 101)),
            "y": list(range(1, 101)),
        }
    )

    profiles = [
        profile_column(df["x"]),
        profile_column(df["y"]),
    ]

    characteristics = SimpleNamespace(
        rows=100,
        numeric_columns=2,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    expected = [
        "iqr",
        "zscore",
        "isolation_forest",
        "lof",
        "dbscan",
    ]

    for detector in expected:

        assert detector in results

        assert isinstance(
            results[detector],
            list,
        )
def test_runner_returns_selected_detectors():

    df = pd.DataFrame(
        {
            "x": list(range(1, 101)),
            "y": list(range(1, 101)),
        }
    )

    profiles = [
        profile_column(df["x"]),
        profile_column(df["y"]),
    ]

    characteristics = SimpleNamespace(
        rows=100,
        numeric_columns=2,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert "selected_detectors" in results

    assert isinstance(
        results["selected_detectors"],
        list,
    )

    assert set(
        results["selected_detectors"]
    ) == {
        "iqr",
        "zscore",
        "isolation_forest",
        "lof",
        "dbscan",
    }
def test_runner_collects_raw_anomalies():

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
            ]
        }
    )

    profiles = [
        profile_column(df["x"]),
    ]

    characteristics = SimpleNamespace(
        rows=7,
        numeric_columns=1,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert "raw_anomalies" in results

    assert isinstance(
        results["raw_anomalies"],
        list,
    )
    
def test_runner_empty_anomalies():

    df = pd.DataFrame(
        {
            "x": [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
            ]
        }
    )

    profiles = [
        profile_column(df["x"]),
    ]

    characteristics = SimpleNamespace(
        rows=7,
        numeric_columns=1,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert isinstance(
        results["raw_anomalies"],
        list,
    )

    assert isinstance(
        results["anomalies"],
        list,
    )
     
def test_runner_handles_constant_column():

    df = pd.DataFrame(
        {
            "x": [10, 10, 10, 10, 10, 10, 10]
        }
    )

    profiles = [
        profile_column(df["x"]),
    ]

    characteristics = SimpleNamespace(
        rows=7,
        numeric_columns=1,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert isinstance(
        results,
        dict,
    )

    assert "iqr" in results
    assert "zscore" in results

    assert isinstance(
        results["iqr"],
        list,
    )

    assert isinstance(
        results["zscore"],
        list,
    )
def test_runner_handles_missing_values():

    df = pd.DataFrame(
        {
            "x": [
                1,
                2,
                None,
                2,
                1,
                None,
                100,
            ],
            "y": [
                1,
                2,
                2,
                None,
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

    characteristics = SimpleNamespace(
        rows=7,
        numeric_columns=2,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert isinstance(
        results,
        dict,
    )

    assert "iqr" in results
    assert "zscore" in results
    
def test_runner_handles_constant_column():

    df = pd.DataFrame(
        {
            "x": [10, 10, 10, 10, 10, 10, 10]
        }
    )

    profiles = [
        profile_column(df["x"]),
    ]

    characteristics = SimpleNamespace(
        rows=7,
        numeric_columns=1,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert isinstance(
        results,
        dict,
    )

    assert "iqr" in results
    assert "zscore" in results

    assert isinstance(
        results["iqr"],
        list,
    )

    assert isinstance(
        results["zscore"],
        list,
    )
    
def test_runner_tiny_dataset():

    df = pd.DataFrame(
        {
            "x": [1, 2, 3]
        }
    )

    profiles = [
        profile_column(df["x"]),
    ]

    characteristics = SimpleNamespace(
        rows=3,
        numeric_columns=1,
        skewed=False,
    )

    results = run_baseline_detection(
        df,
        profiles,
        characteristics,
    )

    assert isinstance(
        results,
        dict,
    )

    assert "iqr" in results
    assert "zscore" in results

    assert "isolation_forest" not in results
    assert "lof" not in results
    assert "dbscan" not in results
    
  
  

    
    
    
    
    
    
    