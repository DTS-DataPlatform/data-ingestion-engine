from dataclasses import dataclass, field
import pandas as pd


@dataclass
class IngestionResult:
    df: pd.DataFrame
    warnings: list[dict] = field(default_factory=list)