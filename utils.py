import polars as pl


def load_data(filepath):
    return pl.read_csv(filepath, separator=';')