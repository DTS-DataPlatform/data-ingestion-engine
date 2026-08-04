import json
import pandas as pd


def read_json_file(file_path: str) -> pd.DataFrame:

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    df = pd.json_normalize(data)

    return df