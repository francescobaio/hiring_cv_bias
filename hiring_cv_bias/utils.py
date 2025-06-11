from typing import Dict, List

import polars as pl


def load_data(filepath):
    return pl.read_csv(filepath, separator=";")


def load_excel_sheets(path: str, sheets: List[str]) -> Dict[str, pl.DataFrame]:
    return {sheet: pl.read_excel(path, sheet_name=sheet) for sheet in sheets}


def localize(latitude: float) -> str:
    if latitude > 44.5:
        return "NORTH"
    if latitude < 40:
        return "SOUTH"
    return "CENTER"
