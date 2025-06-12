import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from hiring_cv_bias.exploration.utils import compute_top_disparity_values

EPSILON = sys.float_info.epsilon


def plot_histogram(
    column: pl.Series,
    ax: Optional[matplotlib.axes.Axes] = None,
    title: str = "",
    normalize: bool = False,
    top_n: int = 10,
    sort=True,
    custom_colors: Optional[Dict[str, str]] = None,
    use_only_suffixes: bool = False,
    x_labels_rotation: int = 70,
    weights_dict: Optional[Dict[str, float]] = None,
) -> None:
    frequencies = column.value_counts(
        normalize=normalize, name="frequency", sort=sort
    ).head(top_n)

    if weights_dict is not None:
        frequencies_with_weights = frequencies.with_columns(
            frequencies[column.name]
            .str.split("_")
            .list.last()
            .replace_strict(weights_dict)
            .alias("weight")
        )
        frequencies = frequencies_with_weights.with_columns(
            pl.col("frequency") * pl.col("weight").alias("frequency")
        ).drop("weight")
        frequencies = frequencies.with_columns(
            pl.col("frequency") / pl.col("frequency").sum().alias("frequency")
        )
        ordering_dict = {key: idx for idx, key in enumerate(list(weights_dict.keys()))}
        frequencies = frequencies.sort(
            pl.col(column.name).str.split("_").list.last().replace(ordering_dict)
        )

    if ax is None:
        _, ax = plt.subplots()
    if use_only_suffixes:
        x_labels = np.array(frequencies[column.name].str.split("_").to_list())[:, 1]
    else:
        x_labels = frequencies[column.name].to_numpy()

    ax.set_xticks(
        ticks=[*range(0, len(frequencies[column.name]))],
        labels=x_labels,
        rotation=x_labels_rotation,
    )
    ax.set_title(title)
    if custom_colors is not None:
        colors: List[str] = []
        for label in frequencies[column.name]:
            for suffix, color in custom_colors.items():
                if label.endswith(suffix):
                    colors.append(color)
                    break
    else:
        colors = ["#1f77b4"]
    if not normalize and frequencies["frequency"].dtype.is_integer():
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.bar(
        frequencies[column.name],
        frequencies["frequency"],
        color=colors,
        edgecolor="black",
    )


def compute_skills_frequency(cv_skills: pl.DataFrame, column_name: str) -> pl.DataFrame:
    """ """
    skills_frequency = (
        cv_skills.filter(cv_skills[column_name].is_not_null())
        .group_by(column_name)
        .agg(pl.count().alias("Frequency"))
        .sort("Frequency", descending=True)
    )

    return skills_frequency


def plot_frequency(
    data: pl.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    orientation: str = "v",
    top_n: int = 10,
):
    data_pd = data.head(top_n).to_pandas()

    plt.figure(figsize=(7, 6))
    if orientation != "h":
        sns.barplot(
            data=data_pd, x=x_col, y=y_col, hue=y_col, palette="Blues_r", edgecolor="k"
        )
    else:
        sns.barplot(
            data=data_pd, x=y_col, y=x_col, hue=y_col, palette="Blues_r", edgecolor="k"
        )

    plt.xlabel(x_col if orientation == "v" else y_col)
    plt.ylabel(y_col if orientation == "v" else x_col)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_skills_frequency(cv_skills: pl.DataFrame) -> None:
    """ """
    skills_frequency = compute_skills_frequency(cv_skills, "Skill_Type")
    plot_frequency(
        skills_frequency,
        "Frequency",
        "Skill_Type",
        "Histogram of Skill Type Frequencies",
        orientation="h",
    )


def plot_skills_per_category(
    cv_skills: pl.DataFrame, skill_type: str, top_n: int = 5
) -> pl.DataFrame:
    """ """
    filtered_data = cv_skills.filter(cv_skills["Skill_Type"] == skill_type)
    skills_frequency = compute_skills_frequency(filtered_data, "Skill")
    plot_frequency(
        skills_frequency,
        "Frequency",
        "Skill",
        f"Top {top_n} skills in {skill_type}",
        top_n=top_n,
    )
    return skills_frequency


def plot_top_skills_for_job_title(
    cv_skills: pl.DataFrame, job_title: str, type_skill: str, top_n: int = 10
):
    """"""
    job_titles_df = (
        cv_skills.filter(pl.col("Skill_Type") == "Job_title")
        .select(["CANDIDATE_ID", "Skill"])
        .rename({"Skill": "Job_title"})
    )
    candidate_skills = cv_skills.filter(pl.col("Skill_Type") == type_skill).select(
        ["CANDIDATE_ID", "Skill"]
    )

    job_title_skills = job_titles_df.join(
        candidate_skills, on="CANDIDATE_ID", how="inner"
    )
    job_skill_frequency = (
        job_title_skills.group_by(["Job_title", "Skill"])
        .agg(pl.count().alias("Frequency"))
        .sort("Frequency", descending=True)
    )

    candidate_skills_filtered = job_skill_frequency.filter(
        pl.col("Job_title") == job_title
    )
    plot_frequency(
        candidate_skills_filtered,
        "Frequency",
        "Skill",
        f"Top {top_n} {type_skill} skills for {job_title}",
        top_n=top_n,
    )


def compute_and_plot_disparity(
    columns: Dict[str, pl.Series],
    min_threshold: int = 25,
    use_percentiles: bool = True,
    colors: Optional[Dict[str, str]] = None,
    attribute_name: str = "",
    weights_dict: Optional[Dict[str, float]] = None,
    top_n: int = 5,
    fig_size: Tuple[int, int] = (14, 8),
):
    if weights_dict is not None:
        weights = list(weights_dict.values())
    else:
        weights_dict = {attr: 1 for attr in columns}
        weights = None

    if colors is None:
        colors = {attr: "#1f77b4" for attr in columns}
    top_values, top_disparities = compute_top_disparity_values(
        columns=list(columns.values()),
        weights=weights,
        min_threshold=min_threshold,
        use_percentiles=use_percentiles,
        top_n=top_n,
    )

    if len(top_values) < top_n:
        top_n = len(top_values)

    percs: Dict[str, List[float]] = defaultdict(list)
    for value in top_values:
        raw_counts = {
            attr: (len(column.filter(column == value)) * weights_dict[attr]) + EPSILON
            for attr, column in columns.items()
        }
        normalization_factor = sum(list(raw_counts.values()))
        for attr, count in raw_counts.items():
            percs[attr].append((count / normalization_factor))

    x = np.arange(top_n)
    num_attributes = len(columns)
    width = 0.7 / num_attributes
    _, ax = plt.subplots(figsize=fig_size)

    for i, (attr, perc) in enumerate(percs.items()):
        bar_position = x + (i - (num_attributes - 1) / 2) * width
        ax.bar(
            bar_position,
            np.array(perc) * 100,
            width,
            label=attr,
            color=colors[attr],
            edgecolor="k",
        )

    for i in range(top_n):
        max_height = max([perc[i] * 100 for perc in percs.values()])
        ax.text(
            x[i],
            max_height + 5,
            rf"$\mathbf{{Gini\ Index:\ {top_disparities[i]:.2f}}}$",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        ax.add_patch(
            plt.Rectangle(
                (x[i] - 0.3, max_height + 1),
                0.6,
                4,
                facecolor="none",
                edgecolor="none",
                label="_nolegend_",
            )
        )
        for j, (_, perc) in enumerate(percs.items()):
            bar_position = x[i] + (j - (num_attributes - 1) / 2) * width
            ax.text(
                bar_position,
                perc[i] * 100 + 1,
                f"{perc[i] * 100:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Percentage of Skill Holders", fontsize=12)
    ax.set_title(f"{attribute_name} with highest disparity", fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(top_values, rotation=60, ha="right", fontsize=10)

    max_bar_height = max([max(perc) * 100 for perc in percs.values()])

    ax.set_ylim(0, max_bar_height * 1.25)
    ax.legend(
        loc="best",
        fontsize=10,
        frameon=False,
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


def plot_target_distribution(
    dfs_per_attribute: Dict[str, pl.DataFrame],
    title: str,
    group_by: str = "Skill_Type",
    normalize=True,
):
    fig, axs = plt.subplots(1, len(dfs_per_attribute), sharex=True, sharey=True)
    fig.set_size_inches(20, 5)
    fig.suptitle(title, fontsize=15, y=1.05)

    for idx, key in enumerate(dfs_per_attribute):
        axs[idx].tick_params(labelleft=True, labelsize="large")
        plot_histogram(
            dfs_per_attribute[key][group_by],
            ax=axs[idx],
            title=key,
            normalize=normalize,
        )
