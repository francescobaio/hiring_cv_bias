import re
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import polars as pl
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag


def plot_distribution_bar(
    df: pl.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, title: str
) -> None:
    x = df[x_col].to_list()
    y = df[y_col].to_list()

    plt.figure(figsize=(8, 5))
    plt.bar(x, y, edgecolor="k")
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_boxplot(
    data: Union[Sequence, pl.Series, Sequence[Sequence]],
    labels: Optional[Sequence[str]] = None,
    title: str = "",
    xlabel: str = "",
    figsize: Tuple[int, int] = (8, 3),
    colors: Optional[Sequence[str]] = None,
    vert: bool = False,
    show_grid: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    box = ax.boxplot(
        data,
        vert=vert,
        patch_artist=True,
        labels=labels,
        boxprops=dict(edgecolor="black"),
        medianprops=dict(color="red"),
    )

    if colors:
        for patch, c in zip(box["boxes"], colors):
            patch.set_facecolor(c)
    else:
        for patch in box["boxes"]:
            patch.set_facecolor("skyblue")

    ax.set_title(title, pad=10)
    if vert:
        ax.set_ylabel(xlabel)
    else:
        ax.set_xlabel(xlabel)
    if show_grid:
        ax.grid(axis="y" if vert else "x", linestyle="--", alpha=0.7)

    fig.tight_layout()
    plt.show()


def extract_gender_from_zippia(
    url: str,
) -> Union[Tuple[float, float], Tuple[None, None]]:
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        container = soup.find("div", string=re.compile(r"gender ratio", re.I))
        if not container:
            return None, None

        parent_div = container.find_parent("div")
        if isinstance(parent_div, Tag):
            value_div = parent_div.find_all("div")[-1]

        if not value_div:
            return None, None

        text = value_div.get_text(separator="\n").strip()
        male_match = re.search(r"Male\s*[--]\s*(\d+)%", text)
        female_match = re.search(r"Female\s*[--]\s*(\d+)%", text)

        if male_match and female_match:
            male = float(male_match.group(1))
            female = float(female_match.group(1))
            return male, female

        return None, None

    except Exception as e:
        print(f"Errore su {url}: {e}")
        return None, None


def add_suffix_and_concat(columns: List[pl.Series], suffixes: List[str]):
    columns_with_suffix = [column + suffix for column, suffix in zip(columns, suffixes)]
    concat_columns = pl.concat(columns_with_suffix)
    return concat_columns


def split_df_per_attribute(
    df: pl.DataFrame, attribute_name: str
) -> Dict[str, pl.DataFrame]:
    dfs_per_attribute = {}
    for value in df[attribute_name].unique(maintain_order=True):
        filtered_df = df.filter(pl.col(attribute_name) == value)
        dfs_per_attribute[value] = filtered_df
    return dfs_per_attribute
