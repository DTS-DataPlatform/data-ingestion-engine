import pandas as pd

from app.core.table import UnifiedTable

from .models import DatasetProfile
from .column_profiler import profile_column


def build_numeric_dataframe(
    df,
    profiles
):

    numeric_data = {}

    excluded_types = {
        "ID",
        "PHONE",
        "EMAIL",
        "DATE",
    }

    for profile in profiles:

        if profile.semantic_type in excluded_types:
            continue

        numeric = pd.to_numeric(
            df[profile.name],
            errors="coerce"
        )

        numeric_ratio = (
            numeric.notna().mean()
        )

        if numeric_ratio >= 0.8:

            numeric_data[
                profile.name
            ] = numeric

    return pd.DataFrame(
        numeric_data
    )


def profile_dataset(
    table: UnifiedTable
) -> DatasetProfile:

    df = table.dataframe

    # =========================
    # Column profiling
    # =========================

    profiles = []

    for column in df.columns:

        profile = profile_column(
            df[column]
        )

        profiles.append(profile)

    # =========================
    # Numeric dataframe
    # =========================

    numeric_df = build_numeric_dataframe(
        df,
        profiles
    )

    # =========================
    # Correlation
    # =========================

    correlation = None

    if numeric_df.shape[1] >= 2:

        correlation = (
            numeric_df
            .corr()
            .to_dict()
        )

    # =========================
    # Dataset profile
    # =========================

    return DatasetProfile(
        rows=table.rows,
        columns=table.columns,
        column_profiles=profiles,
        correlation=correlation,
    )