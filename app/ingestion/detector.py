import pandas as pd
from typing import Any

def infer_schema(df: pd.DataFrame) -> list[dict[str, Any]]:
    schema = []
    
    for column in df.columns:
        series = df[column]
        
        total = len(series)
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        
        schema.append({
            "name": str(column),
            "dtype": str(series.dtype),
            
            "nullable": missing_count > 0,
            "missing_count": missing_count,
            "missing_ratio": missing_count / total if total > 0 else 0.0,
            
            "unique_count": unique_count,
            "unique_ratio": unique_count / total if total > 0 else 0.0,
            
            "sample_values": series.dropna()
                .head(5)
                .tolist()
        })
    return schema