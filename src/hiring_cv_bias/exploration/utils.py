import re
from itertools import combinations
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests
from bs4 import BeautifulSoup


def localize(latitude: float) -> str:
    if latitude > 44.5:
        return "NORTH"
    if latitude < 42:
        return "SOUTH"
    return "CENTER"


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


def extract_gender_from_zippia(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        container = soup.find("div", string=re.compile(r"gender ratio", re.I))
        if not container:
            return None, None

        parent_div = container.find_parent("div")
        value_div = parent_div.find_all("div")[-1]

        if not value_div:
            return None, None

        text = value_div.get_text(separator="\n").strip()
        male_match = re.search(r"Male\s*[–-]\s*(\d+)%", text)
        female_match = re.search(r"Female\s*[–-]\s*(\d+)%", text)

        if male_match and female_match:
            male = float(male_match.group(1))
            female = float(female_match.group(1))
            return male, female

        return None, None

    except Exception as e:
        print(f"Errore su {url}: {e}")
        return None, None


def compute_max_disparity_value(
    columns: List[pl.Series],
    weights: List[float] = None,
    min_threshold: int = 25,
    use_percentiles: bool = True,
):
    if weights is None:
        weights = [1] * len(columns)
    first_set = set(columns[0].drop_nulls())
    other_sets = [set(column.drop_nulls()) for column in columns]
    common_values = first_set.intersection(*other_sets)

    counts_per_column = [
        column.filter(column.is_in(common_values)).value_counts() for column in columns
    ]
    total_counts = (
        pl.concat(counts_per_column)
        .group_by(counts_per_column[0].columns[0])
        .agg(pl.col("count").sum())
    )
    if use_percentiles:
        min_count = np.percentile(total_counts["count"], min_threshold)
    else:
        min_count = min_threshold

    common_values_filtered = total_counts.filter(pl.col("count") >= min_count)[
        total_counts.columns[0]
    ]
    counts_per_column_filtered = [
        column.filter(column[column.columns[0]].is_in(common_values_filtered))
        for column in counts_per_column
    ]

    max_disparity = 0
    for value in common_values_filtered.to_list():
        counts = [
            filtered_count.filter(filtered_count[filtered_count.columns[0]] == value)[
                "count"
            ].item()
            for filtered_count in counts_per_column_filtered
        ]
        ratios = [
            (count / sum(counts)) * weight for count, weight in zip(counts, weights)
        ]

        current_disparity = compute_disparity(ratios)

        if current_disparity > max_disparity:
            max_disparity = current_disparity
            max_disparity_value = value
    return max_disparity_value, max_disparity


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
