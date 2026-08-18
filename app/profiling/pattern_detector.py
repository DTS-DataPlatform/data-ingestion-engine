import re
import pandas as pd


EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
)

PHONE_PATTERN = re.compile(
    r"^\+?\d{9,15}$"
)

DATE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}$"
)


def detect_pattern(series: pd.Series) -> dict:

    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    if values.empty:
        return {
            "type": "unknown",
            "match_ratio": 0.0
        }

    email_ratio = values.apply(
        lambda x: bool(
            EMAIL_PATTERN.match(x)
        )
    ).mean()

    phone_ratio = values.apply(
        lambda x: bool(
            PHONE_PATTERN.match(x)
        )
    ).mean()

    date_ratio = values.apply(
        lambda x: bool(
            DATE_PATTERN.match(x)
        )
    ).mean()

    ratios = {
        "EMAIL_LIKE": float(email_ratio),
        "PHONE_LIKE": float(phone_ratio),
        "DATE_LIKE": float(date_ratio),
    }

    pattern_type = max(
        ratios,
        key=ratios.get
    )

    match_ratio = ratios[pattern_type]

    if match_ratio >= 0.8:

        return {
            "type": pattern_type,
            "match_ratio": match_ratio
        }

    return {
        "type": "unknown",
        "match_ratio": match_ratio
    }