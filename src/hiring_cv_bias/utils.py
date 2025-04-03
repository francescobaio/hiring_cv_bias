import polars as pl
from typing import List, Dict


def load_data(filepath):
    return pl.read_csv(filepath, separator=";")


def load_excel_sheets(path: str, sheets: List[str]) -> Dict[str, pl.DataFrame]:
    return {sheet: pl.read_excel(path, sheet_name=sheet) for sheet in sheets}
