import csv


def detect_malformed_rows(
    file_path: str,
    encoding: str,
    separator: str,
    expected_fields: int
) -> list[dict]:

    warnings = []

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

        for row_number, row in enumerate(reader, start=1):

            actual_fields = len(row)

            if actual_fields != expected_fields:

                warnings.append({
                    "type": "malformed_row",
                    "row": row_number,
                    "expected_fields": expected_fields,
                    "actual_fields": actual_fields,
                    "message": (
                        f"Expected {expected_fields} fields, "
                        f"found {actual_fields}"
                    )
                })

    return warnings