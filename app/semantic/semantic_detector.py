import re

from app.profiling.models import ColumnProfile


SEMANTIC_RULES = {

    "AGE": [
        "age",
        "age_years",
        "years_old",
    ],

    "MONEY": [
        "salary",
        "income",
        "wage",
        "pay",
        "price",
        "cost",
        "amount",
    ],

    "PHONE": [
        "phone",
        "mobile",
        "phone_number",
        "mobile_number",
        "contact_number",
    ],

    "EMAIL": [
        "email",
        "email_address",
        "mail",
    ],

    "DATE": [
        "date",
        "birth_date",
        "birthday",
        "dob",
        "date_of_birth",
    ],

    "ID": [
        "id",
        "identifier",
        "code",
    ],
}


def normalize_column_name(
    name: str
) -> str:

    name = name.lower().strip()

    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        name
    )

    return name.strip("_")


def detect_semantic_type(
    profile: ColumnProfile
) -> tuple[str | None, float, list[str]]:

    column_name = normalize_column_name(
        profile.name
    )

    scores = {}
    evidences = {}

    # ======================================
    # 1. COLUMN NAME EVIDENCE
    # ======================================

    name_tokens = set(
        column_name.split("_")
    )

    for semantic_type, keywords in SEMANTIC_RULES.items():

        for keyword in keywords:

            keyword_tokens = set(
                keyword.split("_")
            )

            if (
                column_name == keyword
                or keyword in column_name
                or keyword_tokens.issubset(name_tokens)
            ):

                scores[semantic_type] = (
                    scores.get(
                        semantic_type,
                        0
                    ) + 0.50
                )

                evidences.setdefault(
                    semantic_type,
                    []
                ).append(
                    f"column_name contains '{keyword}'"
                )

                break

    # ======================================
    # 2. PATTERN EVIDENCE
    # ======================================

    pattern = profile.pattern or {}

    pattern_type = pattern.get("type")
    match_ratio = pattern.get(
        "match_ratio",
        0.0
    )

    pattern_mapping = {

        "EMAIL_LIKE": "EMAIL",
        "PHONE_LIKE": "PHONE",
        "DATE_LIKE": "DATE",
    }

    if pattern_type in pattern_mapping:

        semantic_type = pattern_mapping[
            pattern_type
        ]

        scores[semantic_type] = (
            scores.get(
                semantic_type,
                0
            ) + 0.40 * match_ratio
        )

        evidences.setdefault(
            semantic_type,
            []
        ).append(
            f"pattern={pattern_type}"
        )

        evidences[
            semantic_type
        ].append(
            f"pattern_match_ratio={match_ratio:.3f}"
        )

    # ======================================
    # 3. NUMERIC EVIDENCE
    # ======================================

    numeric_ratio = (
        profile.numeric_ratio
        if profile.numeric_ratio is not None
        else 0.0
    )

    numeric_types = {
        "AGE",
        "MONEY",
    }

    for semantic_type in numeric_types:

        if semantic_type in scores:

            if numeric_ratio >= 0.8:

                scores[semantic_type] += 0.30

                evidences[
                    semantic_type
                ].append(
                    f"numeric_ratio={numeric_ratio:.3f}"
                )

    # ======================================
    # 4. NO EVIDENCE
    # ======================================

    if not scores:

        return (
            None,
            0.0,
            []
        )

    # ======================================
    # 5. BEST SEMANTIC TYPE
    # ======================================

    best_type = max(
        scores,
        key=scores.get
    )

    raw_score = scores[best_type]

    confidence = min(
        1.0,
        raw_score
    )

    return (
        best_type,
        confidence,
        evidences[best_type]
    )
    