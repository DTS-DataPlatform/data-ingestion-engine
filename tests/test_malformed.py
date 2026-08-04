from pathlib import Path

from app.ingestion.csv_reader import detect_header, detect_malformed
from app.utils.malformed_detector import detect_malformed_rows


result = detect_malformed("tests/malformed_detector/test1.csv")
print("DATA:")
print(result.df)


print("\nWARNINGS:")

for warning in result.warnings:
    print(warning)
