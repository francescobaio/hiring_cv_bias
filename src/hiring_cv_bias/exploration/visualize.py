import matplotlib
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt


def plot_histogram(
    column: pl.Series,
    ax: matplotlib.axes.Axes = None,
    title: str = "",
    normalize: bool = False,
    top_n: int = 10,
    sort=True,
) -> None:
    frequencies = column.value_counts(
        normalize=normalize, name="frequency", sort=sort
    ).head(top_n)
    if ax is None:
        _, ax = plt.subplots()

    ax.set_xticks(
        ticks=[*range(0, len(frequencies[column.name]))],
        labels=frequencies[column.name],
        rotation=70,
    )
    ax.set_title(title)
    ax.bar(frequencies[column.name], frequencies["frequency"])


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

    plt.figure(figsize=(8, 6))
    if orientation != "h":
        sns.barplot(data=data_pd, x=x_col, y=y_col, hue=y_col, palette="Blues_r")
    else:
        sns.barplot(data=data_pd, x=y_col, y=x_col, hue=y_col, palette="Blues_r")
        # plt.xticks(rotation=45)

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
    """ """
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
