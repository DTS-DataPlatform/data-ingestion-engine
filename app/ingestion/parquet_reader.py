import pandas as pd
import pyarrow.parquet as pq


def read_parquet_file(file_path: str) -> pd.DataFrame:

    df = pd.read_parquet(file_path)

    return df


def get_parquet_schema(file_path: str):

    table = pq.ParquetFile(file_path)

    return table.schema_arrow