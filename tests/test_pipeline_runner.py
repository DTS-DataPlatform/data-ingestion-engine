import pandas as pd

from app.core.table import UnifiedTable
from app.pipeline.runner import run_pipeline


def test_run_pipeline():

    df = pd.DataFrame(
        {
            "customer_id": [
                1,
                2,
                3,
                4,
                5,
            ],
            "age": [
                20,
                25,
                None,
                30,
                999,
            ],
            "salary": [
                10000000,
                12000000,
                None,
                15000000,
                100000000,
            ],
            "city": [
                "Da Nang",
                "Hue",
                None,
                "Da Nang",
                "Da Nang",
            ],
        }
    )

    table = UnifiedTable(
        dataframe=df,
        source_file="test.csv",
        file_type="csv",
    )

    result = run_pipeline(table)

    assert result is not None

    assert result.dataframe is not None

    assert result.dataset_profile is not None
    
    assert result.quality_comparison is not None


    comparison = result.quality_comparison

    assert comparison.before is not None

    assert comparison.after is not None

    assert comparison.quality_improved is True
    
    assert isinstance(
        result.anomalies,
        list,
    )

    assert isinstance(
        result.cleaning_results,
        list,
    )

    assert isinstance(
        result.imputation_results,
        list,
    )

    assert result.quality_report is not None