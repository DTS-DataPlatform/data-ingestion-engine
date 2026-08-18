from pathlib import Path
from uuid import uuid4
from datetime import datetime

from .csv_reader import read_csv_file
from .excel_reader import read_excel_file
from .json_reader import read_json_file
from .parquet_reader import read_parquet_file
from .detector import infer_schema
from .models import DatasetObject
from app.core.table import UnifiedTable


def ingest_file(file_path: str, sheet_name: str | None = None):

    path = Path(file_path)

    extension = path.suffix.lower()

    if extension == ".csv":
        df = read_csv_file(file_path)

    elif extension == ".xlsx":
        if sheet_name is None:
            raise ValueError(
                "sheet_name is required for Excel files"
            )

        df = read_excel_file(
            file_path,
            sheet_name
        )

    elif extension == ".json":
        df = read_json_file(file_path)

    elif extension == ".parquet":
        df = read_parquet_file(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    schema = infer_schema(df)

    dataset = DatasetObject(
        dataset_id=str(uuid4()),
        file_name=path.name,
        file_type=extension.replace(".", ""),
        rows=len(df),
        columns=len(df.columns),
        schema=schema,
        storage_path=str(path),
        created_at=datetime.now(),
    )
    
    table = UnifiedTable(
    dataframe=df,
    source_file=path.name,
    file_type=extension.replace(".", "")
)

    return dataset, table