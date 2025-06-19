import itertools
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from hiring_cv_bias.config import JOB_LINKS
from hiring_cv_bias.exploration.utils import extract_gender_from_zippia


def get_category_distribution(
    df: pl.DataFrame, col: str, count_col_name: str = "count"
) -> pl.DataFrame:
    counts = df.group_by(col).agg(pl.len().alias(count_col_name))
    total = counts.select(pl.col(count_col_name).sum()).item()

    counts = counts.with_columns(
        (pl.col(count_col_name) / total * 100).round(1).alias("percentage")
    ).sort(count_col_name, descending=True)

    return counts


def get_skill_distribution_by_gender(
    df: pl.DataFrame, skill_col: str = "Skill", gender_col: str = "Gender"
) -> pl.DataFrame:
    counts = df.group_by([gender_col, skill_col]).agg(pl.len().alias("count"))
    totals = counts.group_by(gender_col).agg(
        pl.col("count").sum().alias("total_skills")
    )
    result = (
        counts.join(totals, on=gender_col)
        .with_columns(
            (pl.col("count") / pl.col("total_skills") * 100).alias("percentage")
        )
        .sort(["Gender", "percentage"], descending=[False, True])
    )

    return result


def get_skill_target_share(
    df: pl.DataFrame,
    counts_df: pl.DataFrame,
    target_col: str,
    target_values: List[Any],
    skill_col: List[str] = ["Skill_Type"],
) -> pl.DataFrame:
    counts_df = counts_df.with_columns(pl.col("percentage") / 100)

    target_value_counts = [
        (
            df.filter(pl.col(target_col) == value)
            .group_by(skill_col, maintain_order=True)
            .agg(pl.len().alias(f"count_{value.lower()}"))
        ).drop_nulls(skill_col)
        for value in target_values
    ]

    total_counts = target_value_counts[0]
    for df in target_value_counts[1:]:
        total_counts = total_counts.join(df, on=skill_col, how="full", coalesce=True)
    total_counts = total_counts.fill_null(0)

    normalize_factor = total_counts.select(
        pl.sum_horizontal([f"count_{value.lower()}" for value in target_values])
    ).to_series()

    total_counts = total_counts.with_columns(
        [
            pl.col(f"count_{value.lower()}")
            * (1 - (counts_df.filter(pl.col(target_col) == value)["percentage"].item()))
            for value in target_values
        ]
    )

    normalize_factor = normalize_factor / sum(
        [total_counts[f"count_{value.lower()}"] for value in target_values]
    )

    total_counts = total_counts.with_columns(
        [
            (pl.col(f"count_{value.lower()}") * normalize_factor).round().cast(pl.Int64)
            for value in target_values
        ]
    )

    total_counts = total_counts.with_columns(
        pl.sum_horizontal([f"count_{value.lower()}" for value in target_values]).alias(
            "count_total"
        )
    )

    total_counts = total_counts.with_columns(
        [
            (pl.col(f"count_{value.lower()}") / pl.col("count_total") * 100)
            .round(1)
            .alias(f"perc_{value.lower()}")
            for value in target_values
        ]
    )

    target_pairs = list(itertools.combinations(target_values, 2))

    total_counts = total_counts.with_columns(
        pl.mean_horizontal(
            [
                pl.col(f"count_{v1.lower()}") - pl.col(f"count_{v2.lower()}")
                for v1, v2 in target_pairs
            ]
        )
        .cast(pl.Int64)
        .alias("count_diff"),
        pl.mean_horizontal(
            [
                pl.col(f"perc_{v1.lower()}") - pl.col(f"perc_{v2.lower()}")
                for v1, v2 in target_pairs
            ]
        )
        .round(1)
        .alias("perc_diff"),
    )

    return total_counts.sort("count_total", descending=True)


def compute_bias_strenght(
    df: pl.DataFrame,
    counts_df: pl.DataFrame,
    skill_col: List[str] = ["Skill", "Skill_Type"],
    gender_col: str = "Gender",
) -> pl.DataFrame:
    result = get_skill_target_share(
        df,
        counts_df,
        target_col=gender_col,
        target_values=["Male", "Female"],
        skill_col=skill_col,
    )
    result = result.with_columns(
        [
            (
                (
                    pl.col("count_diff")
                    / pl.col("count_total")
                    * pl.col("count_total").log1p()
                ).abs()
            ).alias("bias_strength")
        ]
    )
    return result


def plot_gender_bias_skills_bar(
    df: pl.DataFrame,
    skill_col: str,
    attribute_cols: Dict[str, str],
    bias_col: str,
    title: str,
    top_n: int = 20,
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (14, 6),
):
    if colors is None:
        colors = {key: "#1f77b4" for key in attribute_cols}
    df = df.sort(pl.col(bias_col).abs(), descending=True).head(top_n)

    skills = df[skill_col].to_list()
    percs = {key: df[attr].to_list() for key, attr in attribute_cols.items()}
    x = np.arange(len(skills))
    num_attributes = len(percs)
    width = 0.7 / num_attributes
    _, ax = plt.subplots(figsize=figsize)
    # colors = sns.color_palette("pastel")

    for i, (label, perc) in enumerate(percs.items()):
        bar_position = x + (i - (num_attributes - 1) / 2) * width
        ax.bar(
            bar_position,
            perc,
            width,
            label=label,
            color=colors[label],
            edgecolor="k",
        )

    for i in range(len(skills)):
        for j, (_, perc) in enumerate(percs.items()):
            bar_position = x[i] + (j - (num_attributes - 1) / 2) * width
            ax.text(
                bar_position,
                perc[i] + 1,
                f"{perc[i]:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Percentage of Skill Holders", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(skills, rotation=60, ha="right", fontsize=10)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    plt.show()


def add_zippia_columns(job_df: pl.DataFrame):
    perc_male_zippia, perc_female_zippia = [], []
    for job_link in JOB_LINKS:
        male, female = extract_gender_from_zippia(job_link)
        perc_male_zippia.append(male)
        perc_female_zippia.append(female)

    perc_male_zpa = pl.Series(perc_male_zippia)
    perc_female_zpa = pl.Series(perc_female_zippia)
    job_df = job_df.with_columns(
        [
            perc_female_zpa.alias("perc_female_zippia"),
            perc_male_zpa.alias("perc_male_zippia"),
        ]
    )
    return job_df
