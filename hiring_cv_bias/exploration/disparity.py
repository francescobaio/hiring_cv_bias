from itertools import combinations
from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import zscore


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
