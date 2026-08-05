from app.ingestion.service import ingest_file


files = [
    "storage/data/data.csv",
    "storage/data/data.xlsx",
    "storage/data/data.json",
    "storage/data/test_data.parquet"
]

for file in files:

    dataset, table = ingest_file(
        file,
        sheet_name="Data"
    )

    print("=" * 50)

    print("FILE:", dataset.file_name)
    print("TYPE:", dataset.file_type)
    print("ROWS:", table.rows)
    print("COLUMNS:", table.columns)