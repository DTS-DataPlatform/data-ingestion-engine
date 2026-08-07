import pandas as pd

from .models import ColumnProfile
from .type_utils import (
    is_numeric_column,
    get_numeric_ratio,
)
from .pattern_detector import detect_pattern

from app.semantic.semantic_detector import (
    detect_semantic_type
)


# ============================================================
# CONSTANTS
# ============================================================

# Các semantic type này không nên chạy numerical profiling
# dù dữ liệu bên trong có thể toàn là số.
NON_NUMERICAL_SEMANTIC_TYPES = {
    "ID",
    "PHONE",
    "EMAIL",
    "DATE",
}


# ============================================================
# MAIN COLUMN PROFILER
# ============================================================

def profile_column(
    series: pd.Series
) -> ColumnProfile:

    # ========================================================
    # 1. BASIC INFORMATION
    # ========================================================

    total = len(series)

    # -------------------------
    # Missing
    # -------------------------

    missing_count = int(
        series.isna().sum()
    )

    missing_ratio = (
        missing_count / total
        if total > 0
        else 0.0
    )

    # -------------------------
    # Unique
    # -------------------------

    unique_count = int(
        series.nunique(
            dropna=True
        )
    )

    unique_ratio = (
        unique_count / total
        if total > 0
        else 0.0
    )

    # -------------------------
    # Numeric ratio
    # -------------------------

    numeric_ratio = get_numeric_ratio(
        series
    )

    # ========================================================
    # 2. CREATE INITIAL PROFILE
    # ========================================================

    profile = ColumnProfile(

        name=str(series.name),

        dtype=str(series.dtype),

        semantic_type=None,

        missing_count=missing_count,

        missing_ratio=missing_ratio,

        unique_count=unique_count,

        unique_ratio=unique_ratio,

        numeric_ratio=numeric_ratio,

        pattern=None,

        semantic_confidence=0.0,

        semantic_evidence=[],
    )

    # ========================================================
    # 3. PATTERN DETECTION
    # ========================================================

    profile.pattern = detect_pattern(
        series
    )

    # ========================================================
    # 4. SEMANTIC TYPE DETECTION
    # ========================================================

    semantic_type, confidence, evidence = (
        detect_semantic_type(
            profile
        )
    )

    profile.semantic_type = (
        semantic_type
    )

    profile.semantic_confidence = (
        confidence
    )

    profile.semantic_evidence = (
        evidence
    )

    # ========================================================
    # 5. CHECK WHETHER NUMERICAL PROFILING IS APPROPRIATE
    # ========================================================

    # Trường hợp:
    #
    # age:
    # dtype = object
    # numeric_ratio = 0.98
    #
    # → vẫn phải được numerical profiling.
    #
    # Nhưng:
    #
    # phone:
    # dtype = str
    # numeric_ratio = 0.98
    # semantic_type = PHONE
    #
    # → KHÔNG được numerical profiling.

    semantic_is_non_numeric = (
        profile.semantic_type
        in NON_NUMERICAL_SEMANTIC_TYPES
    )

    numeric_possible = (
        is_numeric_column(series)
        or numeric_ratio >= 0.8
    )

    should_profile_numerically = (
        numeric_possible
        and not semantic_is_non_numeric
    )

    # ========================================================
    # 6. NUMERICAL PROFILING
    # ========================================================

    if should_profile_numerically:

        # Convert sang numeric tạm thời.
        #
        # Không thay đổi DataFrame gốc.
        #
        # Ví dụ:
        #
        # "20"  → 20
        # "21"  → 21
        # "abc" → NaN

        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        )

        # Loại bỏ:
        #
        # NaN
        # invalid numeric values

        clean_series = (
            numeric_series
            .dropna()
        )

        if not clean_series.empty:

            # =================================================
            # MIN
            # =================================================

            profile.min = float(
                clean_series.min()
            )

            # =================================================
            # MAX
            # =================================================

            profile.max = float(
                clean_series.max()
            )

            # =================================================
            # MEAN
            # =================================================

            profile.mean = float(
                clean_series.mean()
            )

            # =================================================
            # MEDIAN
            # =================================================

            profile.median = float(
                clean_series.median()
            )

            # =================================================
            # STANDARD DEVIATION
            # =================================================

            profile.std = float(
                clean_series.std()
            )

            # =================================================
            # QUANTILES
            # =================================================

            profile.quantiles = {

                "q25": float(
                    clean_series.quantile(
                        0.25
                    )
                ),

                "q50": float(
                    clean_series.quantile(
                        0.50
                    )
                ),

                "q75": float(
                    clean_series.quantile(
                        0.75
                    )
                ),
            }

            # =================================================
            # DISTRIBUTION
            # =================================================

            profile.distribution = {

                "skewness": (
                    float(
                        clean_series.skew()
                    )
                ),

                "kurtosis": (

                    float(
                        clean_series.kurt()
                    )

                    if len(clean_series) >= 4

                    else None
                ),
            }

    # ========================================================
    # 7. RETURN
    # ========================================================

    return profile