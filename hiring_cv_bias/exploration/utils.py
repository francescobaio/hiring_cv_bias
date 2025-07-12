import re
from collections.abc import Sequence
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from scipy.stats import zscore


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


def compute_top_disparity_values(
    columns: List[pl.Series],
    weights: Optional[List[float]] = None,
    min_threshold: float = 1.0,
    top_n: int = 5,
) -> Tuple[npt.NDArray, npt.NDArray]:
    if weights is None:
        weights = [1] * len(columns)
    first_set = set(columns[0].drop_nulls())
    other_sets = [set(column.drop_nulls()) for column in columns]
    common_values = first_set.intersection(*other_sets)

    counts_per_column = [
        column.filter(column.is_in(common_values)).value_counts() for column in columns
    ]

    counts_per_column_weighted = [
        column.with_columns(pl.col("count") * weight)
        for column, weight in zip(counts_per_column, weights)
    ]

    total_counts = (
        pl.concat(counts_per_column_weighted)
        .group_by(counts_per_column_weighted[0].columns[0])
        .agg(pl.col("count").sum())
    )

    z_score_thrs = zscore(np.log(total_counts["count"]))

    common_values_filtered = total_counts.filter(z_score_thrs > min_threshold)[
        total_counts.columns[0]
    ]
    counts_per_column_filtered = [
        column.filter(column[column.columns[0]].is_in(common_values_filtered))
        for column in counts_per_column_weighted
    ]

    disparity_values: List[str] = []
    disparities: List[float] = []
    disparity_counts: List[int] = []
    common_values_filtered.sort(in_place=True)

    for value in common_values_filtered:
        counts = [
            filtered_count.filter(filtered_count[filtered_count.columns[0]] == value)[
                "count"
            ].item()
            for filtered_count in counts_per_column_filtered
        ]

        current_disparity = compute_disparity(counts)
        current_disparity_counts = sum(counts)

        disparity_values.append(value)
        disparities.append(current_disparity)
        disparity_counts.append(current_disparity_counts)

    top_values_and_disparities = np.array(
        [
            [value, disparity]
            for value, disparity, _ in sorted(
                zip(disparity_values, disparities, disparity_counts),
                key=lambda triplet: (triplet[1], triplet[2]),
                reverse=True,
            )
        ],
        dtype=object,
    )
    top_values = top_values_and_disparities[:top_n, 0]
    top_disparities = top_values_and_disparities[:top_n, 1]

    return top_values, top_disparities


def compute_disparity(data_points: List[Any]) -> float:
    data_point_pairs = list(combinations(data_points, 2))
    disparity = 0
    for first_data_point, second_data_point in data_point_pairs:
        disparity += abs(first_data_point - second_data_point)
    disparity /= sum(data_points) * len(data_points)
    return disparity


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
