import csv
import re
from typing import Any


# ============================================================
# 1. READ SAMPLE
# ============================================================

def read_sample(
    file_path: str,
    encoding: str = "utf-8",
    separator: str = ",",
    n_rows: int = 10
) -> list[list[str]]:

    rows = []

    with open(
        file_path,
        "r",
        encoding=encoding,
        errors="replace",
        newline=""
    ) as f:

        reader = csv.reader(
            f,
            delimiter=separator
        )

        for row in reader:

            # Bỏ dòng hoàn toàn rỗng
            if not row or all(
                not cell.strip()
                for cell in row
            ):
                continue

            rows.append(row)

            if len(rows) >= n_rows:
                break

    return rows


# ============================================================
# 2. DETECT BASIC DATA TYPE
# ============================================================

def detect_type(value: str) -> str:

    value = value.strip()

    if value == "":
        return "empty"

    # Integer
    if re.fullmatch(
        r"[-+]?\d+",
        value
    ):
        return "int"

    # Float
    if re.fullmatch(
        r"[-+]?\d*\.\d+",
        value
    ):
        return "float"

    return "string"


# ============================================================
# 3. SNIFFER SIGNAL
# ============================================================

def sniffer_signal(
    rows: list[list[str]]
) -> float:

    if len(rows) < 2:
        return 0.5

    sample_lines = [
        ",".join(row)
        for row in rows
    ]

    sample = "\n".join(sample_lines)

    try:

        result = csv.Sniffer().has_header(
            sample
        )

        return 1.0 if result else 0.0

    except csv.Error:

        return 0.5


# ============================================================
# 4. TYPE DIFFERENCE
# ============================================================

def type_difference_score(
    rows: list[list[str]]
) -> float:

    if len(rows) < 2:
        return 0.5

    header_row = rows[0]

    data_rows = rows[1:]

    header_types = [
        detect_type(value)
        for value in header_row
    ]

    differences = 0
    total = 0

    for row in data_rows:

        for header_type, value in zip(
            header_types,
            row
        ):

            data_type = detect_type(value)

            if header_type != data_type:
                differences += 1

            total += 1

    if total == 0:
        return 0.5

    return differences / total


# ============================================================
# 5. TYPE EVIDENCE
# ============================================================

def type_evidence_score(
    type_difference: float
) -> float:

    # Rất khác type
    if type_difference >= 0.6:
        return 1.0

    # Khá khác
    if type_difference >= 0.3:
        return 0.7

    # Có một ít khác biệt
    if type_difference > 0:
        return 0.4

    # Không khác type không có nghĩa là không có header.
    # Đây chỉ là neutral evidence.
    return 0.5


# ============================================================
# 6. COLUMN NAME QUALITY
# ============================================================

def column_name_quality(
    value: str
) -> float:

    value = value.strip()

    if not value:
        return 0.0

    # Numeric
    if re.fullmatch(
        r"[-+]?\d+(\.\d+)?",
        value
    ):
        return 0.0

    # Date
    if re.fullmatch(
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        value
    ):
        return 0.0

    # Không có chữ
    if not re.search(
        r"[A-Za-zÀ-ỹ]",
        value
    ):
        return 0.0

    # Tên kiểu A, B, C, D quá generic
    if len(value) == 1:
        return 0.0

    # Tên rất ngắn
    if len(value) == 2:
        return 0.2

    return 1.0

# ============================================================
# 7. NAME PATTERN SIGNAL
# ============================================================

def name_pattern_signal(
    rows: list[list[str]]
) -> float:

    if not rows:
        return 0.5

    header = rows[0]

    if not header:
        return 0.5

    scores = [
        column_name_quality(value)
        for value in header
    ]

    return sum(scores) / len(scores)


# ============================================================
# 8. DUPLICATE COLUMN SIGNAL
# ============================================================

def duplicate_column_signal(
    header: list[str]
) -> float:

    if not header:
        return 0.0

    normalized = [
        value.strip().lower()
        for value in header
        if value.strip()
    ]

    if not normalized:
        return 0.0

    total = len(normalized)
    unique_count = len(set(normalized))

    duplicate_ratio = (
        total - unique_count
    ) / total

    return max(
        0.0,
        1.0 - duplicate_ratio * 2
    )

# ============================================================
# 9. EMPTY COLUMN SIGNAL
# ============================================================

def empty_column_signal(
    header: list[str]
) -> float:

    if not header:
        return 0.0

    empty_count = sum(
        not value.strip()
        for value in header
    )

    total = len(header)

    empty_ratio = (
        empty_count / total
    )

    return max(
        0.0,
        1.0 - empty_ratio * 2
    )

# ============================================================
# 10. DATA AVAILABILITY SIGNAL
# ============================================================

def data_availability_signal(
    rows: list[list[str]]
) -> float:

    # Chỉ có header candidate
    if len(rows) <= 1:
        return 0.0

    return 1.0

def detect_header_warnings(
    header: list[str]
) -> list[str]:

    warnings = []

    # Empty column
    if any(
        not value.strip()
        for value in header
    ):
        warnings.append(
            "EMPTY_COLUMN_NAME"
        )

    # Duplicate column
    normalized = [
        value.strip().lower()
        for value in header
        if value.strip()
    ]

    if len(normalized) != len(set(normalized)):
        warnings.append(
            "DUPLICATE_COLUMN_NAME"
        )

    return warnings

# ============================================================
# 11. SCORE FUSION
# ============================================================

def calculate_score(
    sniffer: float,
    type_evidence: float,
    name_pattern: float,
    data_available: float
) -> float:

    score = (
        0.20 * sniffer
        + 0.25 * type_evidence
        + 0.45 * name_pattern
        + 0.10 * data_available
    )

    return round(
        max(0.0, min(1.0, score)),
        3
    ) 
# ============================================================
# 12. DECISION
# ============================================================

def make_decision(
    confidence: float,
    data_available: float,
    sniffer: float,
    type_evidence: float,
    name_pattern: float,
    warnings: list[str]
) -> str:

    if data_available == 0:
        return "WARNING"

    # Header name rất rõ ràng
    if (
        name_pattern >= 0.9
        and data_available == 1
        and not warnings
    ):
        return "AUTO"

    # Numeric header: Sniffer có evidence
    # nhưng tên cột không giống tên thông thường
    if (
        sniffer >= 0.8
        and name_pattern < 0.3
    ):
        return "WARNING"

    if warnings:
        return "WARNING"

    if confidence >= 0.80:
        return "AUTO"

    if confidence >= 0.55:
        if (
            sniffer >= 0.5
            or name_pattern >= 0.5
            or type_evidence >= 0.5
        ):
            return "WARNING"

    return "ASK_USER"