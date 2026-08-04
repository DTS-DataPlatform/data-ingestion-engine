from pathlib import Path

from app.ingestion.csv_reader import detect_header


TEST_DIR = Path("tests/header_detection")


test_cases = [
    ("01_clear_header.csv", "AUTO"),
    ("02_no_header.csv", "ASK_USER"),
    ("03_ambiguous.csv", "WARNING"),
    ("04_numeric_header.csv", "WARNING"),
    ("05_string_data.csv", "AUTO"),
    ("06_vietnamese_header.csv", "AUTO"),
    ("07_duplicate_header.csv", "WARNING"),
    ("08_empty_header.csv", "WARNING"),
    ("09_special_header.csv", "AUTO"),
    ("10_long_header.csv", "AUTO"),
    ("11_one_row.csv", "WARNING"),
    ("12_numeric_no_header.csv", "ASK_USER"),
    ("13_header_like_data.csv", "WARNING"),
]


for filename, expected in test_cases:

    path = TEST_DIR / filename

    result = detect_header(str(path))

    actual = result["decision"]

    status = "PASS" if actual == expected else "FAIL"

    print(
        f"{status:5} | "
        f"{filename:30} | "
        f"Expected={expected:10} | "
        f"Actual={actual}"
    )