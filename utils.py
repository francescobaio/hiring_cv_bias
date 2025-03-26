import polars as pl
from typing import List, Dict
import matplotlib.pyplot as plt


def load_data(filepath):
    return pl.read_csv(filepath, separator=";")


def load_excel_sheets(path: str, sheets: List[str]) -> Dict[str, pl.DataFrame]:
    return {sheet: pl.read_excel(path, sheet_name=sheet) for sheet in sheets}


def plot_distribution_bar(
    df: pl.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, title: str
):
    x = df[x_col].to_list()
    y = df[y_col].to_list()

    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
