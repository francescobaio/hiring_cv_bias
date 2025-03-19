import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


def compute_skills_frequency(cv_skills: pl.DataFrame, column_name: str) -> pl.DataFrame:
    """
    Computes the frequency of unique values in a given column.

    Parameters:
    - cv_skills (pl.DataFrame): DataFrame containing skill data.
    - column_name (str): Column to compute frequencies for.

    Returns:
    - pl.DataFrame: DataFrame with value counts sorted in descending order.
    """
    return (
        cv_skills.filter(cv_skills[column_name].is_not_null())
        .group_by(column_name)
        .agg(pl.count().alias("Frequency"))
        .sort("Frequency", descending=True)
    )


def plot_frequency(
    data: pl.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    orientation: str = "h",
    top_n: int = 10,
):
    """
    Plots a frequency distribution as a bar chart.

    Parameters:
    - data (pl.DataFrame): DataFrame containing frequency counts.
    - x_col (str): Column for x-axis.
    - y_col (str): Column for y-axis.
    - title (str): Plot title.
    - orientation (str, optional): "h" for horizontal (default), "v" for vertical.
    - top_n (int, optional): Number of top values to display (default: 10).
    """
    data_pd = data.head(top_n).to_pandas()

    plt.figure(figsize=(10, 8))
    if orientation == "v":
        sns.barplot(data=data_pd, x=x_col, y=y_col, palette="Blues_r")
    else:
        sns.barplot(data=data_pd, x=y_col, y=x_col, palette="Blues_r")
        plt.gca().invert_yaxis()

    plt.xlabel(x_col if orientation == "v" else y_col)
    plt.ylabel(y_col if orientation == "v" else x_col)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_skills_frequency(cv_skills: pl.DataFrame):
    """
    Plots the frequency of different skill types.
    
    Parameters:
    - cv_skills (pl.DataFrame): DataFrame containing skill data.
    """
    skills_frequency = compute_skills_frequency(cv_skills, "Skill_Type")
    plot_frequency(
        skills_frequency,
        "Frequency",
        "Skill_Type",
        "Histogram of Skill Type Frequencies",
        orientation="h",
    )


def plot_skills_per_category(cv_skills: pl.DataFrame, skill_type: str, top_n: int = 5):
    """
    Plots the most common skills for a given skill category.

    Parameters:
    - cv_skills (pl.DataFrame): DataFrame containing skill data.
    - skill_type (str): Skill category to analyze.
    - top_n (int, optional): Number of top skills to display (default: 5).

    Returns:
    - pl.DataFrame: DataFrame with skill frequency counts.
    """
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


def plot_top_skills_for_job_title(cv_skills: pl.DataFrame, job_title: str, top_n: int = 10):
    """
    Plots the most common skills for a specific job title.

    Parameters:
    - cv_skills (pl.DataFrame): DataFrame containing skill data.
    - job_title (str): Job title to analyze.
    - top_n (int, optional): Number of top skills to display (default: 10).
    """
    job_titles_df = (
        cv_skills.filter(pl.col("Skill_Type") == "Job_title")
        .select(["CANDIDATE_ID", "Skill"])
        .rename({"Skill": "Job_title"})
    )
    candidate_skills = cv_skills.filter(pl.col("Skill_Type") != "Job_title").select(["CANDIDATE_ID", "Skill"])

    job_title_skills = job_titles_df.join(candidate_skills, on="CANDIDATE_ID", how="inner")
    job_skill_frequency = (
        job_title_skills.group_by(["Job_title", "Skill"])
        .agg(pl.count().alias("Frequency"))
        .sort("Frequency", descending=True)
    )

    candidate_skills_filtered = job_skill_frequency.filter(pl.col("Job_title") == job_title)
    plot_frequency(
        candidate_skills_filtered,
        "Frequency",
        "Skill",
        f"Top {top_n} skills for {job_title}",
        top_n=top_n,
    )
