from pathlib import Path
from charset_normalizer import from_path
from typing import Any
from app.utils.header_detector import (
    sniffer_signal,
    type_difference_score,
    name_pattern_signal,
    calculate_score,
    make_decision,
    read_sample,
    type_evidence_score,
    duplicate_column_signal,
    empty_column_signal,
    data_availability_signal,
    detect_header_warnings
)
from app.utils.malformed_detector import (
    detect_malformed_rows
)
from app.utils.result import (
    IngestionResult
)

import csv
import pandas as pd

def detect_encoding(file_path: Path) -> str:
    result = from_path(file_path).best()
    
    if result is None:
        return "utf-8"  
    
    return result.encoding

def detect_separator(file_path: Path, encoding: str) -> str:
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        sample = f.read(8192)  # Read the first 8KB for detection
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample,
                                    delimiters=[',', ';', '\t', '|']
                                    )
            return dialect.delimiter
        except csv.Error:
            return ','  
        
def detect_header(
    file_path: str,
    encoding: str = "utf-8",
    separator: str = ",",
    n_rows: int = 10
) -> dict[str, Any]:

    # --------------------------------------------------------
    # Read
    # --------------------------------------------------------

    rows = read_sample(
        file_path=file_path,
        encoding=encoding,
        separator=separator,
        n_rows=n_rows
    )

    # Không có dữ liệu
    if not rows:

        return {
            "has_header": False,
            "confidence": 0.0,
            "decision": "ASK_USER",
            "header_row": None,
            "signals": {}
        }

    # --------------------------------------------------------
    # Candidate header = first row
    # --------------------------------------------------------

    header_row = rows[0]

    # --------------------------------------------------------
    # Signals
    # --------------------------------------------------------

    sniffer_score = sniffer_signal(
        rows
    )

    type_difference = type_difference_score(
        rows
    )

    type_evidence = type_evidence_score(
        type_difference
    )

    name_score = name_pattern_signal(
        rows
    )

    duplicate_score = duplicate_column_signal(
        header_row
    )

    empty_score = empty_column_signal(
        header_row
    )

    data_available = data_availability_signal(
        rows
    )
    
    warnings = detect_header_warnings(
    header_row
)

    # --------------------------------------------------------
    # Score Fusion
    # --------------------------------------------------------

    confidence = calculate_score(

        sniffer=sniffer_score,

        type_evidence=type_evidence,

        name_pattern=name_score,
        
        data_available=data_available
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    decision = make_decision(
        confidence=confidence,
        data_available=data_available,
        sniffer = sniffer_score,
        type_evidence=type_evidence,
        name_pattern=name_score,
        warnings=warnings
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    has_header = (
        decision != "ASK_USER"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "has_header": decision != "ASK_USER",

        "confidence": confidence,

        "decision": decision,

        "header_row": 0,

        "warnings": warnings,

        "signals": {
            "sniffer": round(
                sniffer_score, 3
            ),

            "type_difference": round(
                type_difference, 3
            ),

            "type_evidence": round(
                type_evidence, 3
            ),

            "name_pattern": round(
                name_score, 3
            ),

            "duplicate": round(
                duplicate_score, 3
            ),

            "empty": round(
                empty_score, 3
            ),

            "data_available": round(
                data_available, 3
            )
        }
}
    
def detect_malformed(
    file_path: str,
    encoding: str = "utf-8",
    separator: str = ","
) -> IngestionResult:

    warnings = []

    # -------------------------
    # 1. Đọc header
    # -------------------------

    try:

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

            header = next(reader)

        expected_fields = len(header)

    except Exception as e:

        warnings.append({
            "type": "header_parse_error",
            "row": 1,
            "message": str(e)
        })

        return IngestionResult(
            df=pd.DataFrame(),
            warnings=warnings
        )

    # -------------------------
    # 2. Detect malformed rows
    # -------------------------

    warnings.extend(
        detect_malformed_rows(
            file_path=file_path,
            encoding=encoding,
            separator=separator,
            expected_fields=expected_fields
        )
    )

    # -------------------------
    # 3. Đọc DataFrame
    # -------------------------

    try:

        df = pd.read_csv(
            file_path,
            encoding=encoding,
            sep=separator,
            on_bad_lines="warn"
        )

    except pd.errors.ParserError as e:

        warnings = [
            warning
            for warning in warnings
            if warning["type"] != "malformed_row"
        ]

        warnings.append({
            "type": "parse_error",
            "row": None,
            "message": str(e)
        })

        return IngestionResult(
            df=pd.DataFrame(),
            warnings=warnings
        )

    # -------------------------
    # 4. Return
    # -------------------------

    return IngestionResult(
        df=df,
        warnings=warnings
    )   

def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Đọc CSV và trả về pandas DataFrame.
    """

    # 1. Detect encoding
    encoding = detect_encoding(file_path)

    # 2. Detect separator
    separator = detect_separator(
        file_path,
        encoding
    )

    # 3. Detect header
    has_header = detect_header(
        file_path,
        encoding,
        separator
    )

    # 4. Đọc CSV
    df = pd.read_csv(
        file_path,
        encoding=encoding,
        sep=separator,
        header=0 if has_header else None,
        on_bad_lines="warn"
    )

    return df
    