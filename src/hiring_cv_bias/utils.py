import polars as pl
from typing import Optional

def load_data(filepath: str, separator: str = ';', sheet: Optional[str] = None, load_excel: bool = False) -> pl.DataFrame:
    if load_excel:
        df = pl.read_excel(
            filepath,
            sheet_name=sheet
        )
    else:
        df = pl.read_csv(
            filepath,
            separator=separator,
        )

    return df