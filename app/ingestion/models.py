from dataclasses import dataclass
from typing import Any
from datetime import datetime

@dataclass
class DatasetObject:
    dataset_id: str
    file_name: str
    file_type: str
    
    rows: int
    columns: int
    
    schema: list[dict[str, Any]]
    
    storage_path: str
    created_at: datetime
    