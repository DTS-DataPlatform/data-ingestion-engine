import pandas as pd


def get_excel_sheets(file_path: str):
    excel = pd.ExcelFile(file_path)
    return excel.sheet_names

def read_excel_file(
    file_path: str,
    sheet_name: str
) -> pd.DataFrame:

    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name
    )

    df = df.dropna(how="all")

    return df