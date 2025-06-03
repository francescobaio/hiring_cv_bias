from typing import Dict, List, Optional

import matplotlib
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from hiring_cv_bias.exploration.utils import (
    add_suffix_and_concat,
    compute_max_disparity_value,
)


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


def plot_skills_frequency(cv_skills: pl.DataFrame):
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
    columns: List[pl.Series],
    suffixes: List[str],
    min_threshold: int = 25,
    use_percentiles: bool = True,
    custom_colors: Optional[Dict[str, str]] = None,
    attribute_name: str = "",
    weights_dict: Optional[Dict[str, float]] = None,
    x_label: str = "",
):
    if weights_dict is not None:
        weights = list(weights_dict.values())
    else:
        weights = None

    max_disparity_value, max_disparity = compute_max_disparity_value(
        columns=columns,
        weights=weights,
        min_threshold=min_threshold,
        use_percentiles=use_percentiles,
    )
    columns_filtered = [
        column.filter(column == max_disparity_value) for column in columns
    ]

    total_max_disparity_column = add_suffix_and_concat(
        columns=columns_filtered,
        suffixes=suffixes,
    )

    fig, ax = plt.subplots()
    # ax.text(
    #    0.95,
    #    0.95,
    #    f"Gini Index: {max_disparity: .2f}",
    #    transform=ax.transAxes,
    #    ha="right",
    #    va="top",
    #    fontsize=10,
    #    fontweight="bold",
    # )

    max_disparity_value = max_disparity_value.replace("(m/f)", "")
    fig.suptitle(
        f"{attribute_name} with most disparity: "
        + r"$\mathbf{"
        + max_disparity_value
        + r"}$"
    )
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)
    fig.subplots_adjust(bottom=0.25)

    plot_histogram(
        total_max_disparity_column,
        custom_colors=custom_colors,
        ax=ax,
        use_only_suffixes=True,
        x_labels_rotation=0,
        weights_dict=weights_dict,
    )

    ax.plot([], [], " ", label=rf"$\mathbf{{Gini\ Index:\ {max_disparity:.2f}}}$")

    ax.legend(
        loc="best",
        fontsize=10,
        frameon=False,
        handlelength=0,
        handletextpad=0,
    )


def plot_skill_type_distribution(
    dfs_per_attribute: Dict[str, pl.DataFrame], attribute_name: str
):
    fig, axs = plt.subplots(1, len(dfs_per_attribute), sharex=True, sharey=True)
    fig.set_size_inches(20, 5)
    fig.suptitle(f"{attribute_name} Skill Type Distribution", fontsize=15)

    for idx, key in enumerate(dfs_per_attribute):
        axs[idx].tick_params(labelleft=True, labelsize="large")
        plot_histogram(
            dfs_per_attribute[key]["Skill_Type"], ax=axs[idx], title=key, normalize=True
        )
