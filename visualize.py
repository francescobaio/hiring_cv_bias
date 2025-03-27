import polars as pl
from typing import Optional
from matplotlib import pyplot as plt
import seaborn as sns


def plot_histogram(column: pl.Series, normalize: bool = False, top_n=10) -> None:
    frequencies = column.value_counts(
        normalize=normalize, name="frequency", sort=True
    ).head(top_n)
    plt.xticks(rotation=70)
    plt.bar(frequencies[column.name], frequencies["frequency"])


def plot_correlation_matrix(df: pl.DataFrame) -> None:
    print(df.corr())
    sns.heatmap(df.corr())
