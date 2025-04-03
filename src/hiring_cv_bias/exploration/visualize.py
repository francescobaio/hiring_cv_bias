import polars as pl
from matplotlib import pyplot as plt


def plot_histogram(column: pl.Series, normalize: bool = False, top_n=10) -> None:
    frequencies = column.value_counts(
        normalize=normalize, name="frequency", sort=True
    ).head(top_n)
    plt.xticks(rotation=70)
    plt.bar(frequencies[column.name], frequencies["frequency"])

