from dataclasses import dataclass
import pandas as pd

@dataclass
class UnifiedTable:
    dataframe: pd.DataFrame
    source_file: str
    file_type: str
    
    @property
    def rows(self)->int:
        return len(self.dataframe)
    
    @property
    def columns(self)->int:
        return len(self.dataframe.columns)