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


def get_skill_gender_share(
    df: pl.DataFrame,
    skill_col: str = ["Skill", "Skill_Type"],
    gender_col: str = "Gender",
) -> pl.DataFrame:
    male_counts = (
        df.filter(pl.col(gender_col) == "Male")
        .group_by(skill_col)
        .agg(pl.len().alias("count_male"))
    )

    female_counts = (
        df.filter(pl.col(gender_col) == "Female")
        .group_by(skill_col)
        .agg(pl.len().alias("count_female"))
    )

    result = male_counts.join(female_counts, on=skill_col, how="outer").fill_null(0)
    result = result.with_columns(
        [(pl.col("count_male") + pl.col("count_female")).alias("count_total")]
    )

    result = result.with_columns(
        [
            (pl.col("count_female") / pl.col("count_total") * 100)
            .round(1)
            .alias("perc_female"),
            (pl.col("count_male") / pl.col("count_total") * 100)
            .round(1)
            .alias("perc_male"),
        ]
    )

    result = result.with_columns(
        [
            (pl.col("perc_male") - pl.col("perc_female")).alias("perc_diff"),
            (
                pl.col("count_male").cast(pl.Int64)
                - pl.col("count_female").cast(pl.Int64)
            ).alias("count_diff"),
        ]
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
    male_col: str,
    female_col: str,
    bias_col: str,
    top_n: int = 20,
):
    df = df.sort(bias_col, descending=True).head(top_n)

    skills = df[skill_col].to_list()
    perc_m = df[male_col].to_list()
    perc_f = df[female_col].to_list()
    x = np.arange(len(skills))
    width = 0.35

    _, ax = plt.subplots(figsize=(14, 6))
    # colors = sns.color_palette("pastel")

    ax.bar(x - width / 2, perc_m, width, label="Male", color="skyblue", edgecolor="k")
    ax.bar(
        x + width / 2, perc_f, width, label="Female", color="lightcoral", edgecolor="k"
    )

    for i in range(len(skills)):
        ax.text(
            x[i] - width / 2,
            perc_m[i] + 1,
            f"{perc_m[i]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            x[i] + width / 2,
            perc_f[i] + 1,
            f"{perc_f[i]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Percentage of Skill Holders", fontsize=12)
    ax.set_title(f"Top {top_n} Skills with Highest Gender Imbalance", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(skills, rotation=60, ha="right", fontsize=10)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    plt.show()


def get_skilltype_gender_share(
    df: pl.DataFrame, skilltype_col: str = "Skill_Type", gender_col: str = "Gender"
) -> pl.DataFrame:
    male_counts = (
        df.filter(pl.col(gender_col) == "Male")
        .group_by(skilltype_col)
        .agg(pl.len().alias("count_male"))
    )

    female_counts = (
        df.filter(pl.col(gender_col) == "Female")
        .group_by(skilltype_col)
        .agg(pl.len().alias("count_female"))
    )

    result = male_counts.join(female_counts, on=skilltype_col, how="outer").fill_null(0)

    result = result.with_columns(
        [(pl.col("count_male") + pl.col("count_female")).alias("count_total")]
    )

    result = result.with_columns(
        [
            (pl.col("count_female") / pl.col("count_total") * 100)
            .round(1)
            .alias("perc_female"),
            (pl.col("count_male") / pl.col("count_total") * 100)
            .round(1)
            .alias("perc_male"),
        ]
    )

    result = result.with_columns(
        [
            (
                pl.col("count_male").cast(pl.Int64)
                - pl.col("count_female").cast(pl.Int64)
            ).alias("count_diff"),
            (pl.col("perc_male") - pl.col("perc_female")).alias("perc_diff"),
        ]
    )

    return result.sort("count_total", descending=True)


def add_zippia_columns(job_df: pl.DataFrame):
    perc_male_zippia, perc_female_zippia = [], []
    for job_link in JOB_LINKS:
        male, female = extract_gender_from_zippia(job_link)
        perc_male_zippia.append(male)
        perc_female_zippia.append(female)

    perc_male_zippia = pl.Series(perc_male_zippia)
    perc_female_zippia = pl.Series(perc_female_zippia)
    job_df = job_df.with_columns(
        [
            perc_female_zippia.alias("perc_female_zippia"),
            perc_male_zippia.alias("perc_male_zippia"),
        ]
    )
    return job_df
