from app.ingestion.service import ingest_file


dataset, df = ingest_file("tests/sample.csv")

print(dataset)

print("\nDATA:")
print(df)

print("\nSCHEMA:")
for column in dataset.schema:
    print(column)