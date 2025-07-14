from typing import Dict, List

import polars as pl


def load_data(filepath: str) -> pl.DataFrame:
    return pl.read_csv(filepath, separator=";")


def load_excel_sheets(path: str, sheets: List[str]) -> Dict[str, pl.DataFrame]:
    return {sheet: pl.read_excel(path, sheet_name=sheet) for sheet in sheets}


def filter_unknown_and_other_rows(
    cv_df: pl.DataFrame,
) -> pl.DataFrame:
    cv_df = cv_df.filter(pl.col("Gender").is_in(["Male", "Female"]))
    cv_df = cv_df.filter(pl.col("Age_bucket").is_in(["25-34", "45-54", "55-74"]))
    return cv_df
