import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


def compute_skills_frequency(cv_data: pl.DataFrame, column_name: str) -> pl.DataFrame:
    """ """
    skills_frequency = (
        cv_data.filter(cv_data[column_name].is_not_null())
        .group_by(column_name)
        .agg(pl.count().alias("Frequency"))
        .sort("Frequency", descending=True)
    )

    return skills_frequency


def plot_skills_frequency(cv_data: pl.DataFrame):
    """ """
    skills_frequency = compute_skills_frequency(cv_data, "Skill_Type")
    skills_pd = skills_frequency.to_pandas()

    plt.figure(figsize=(7, 5))
    sns.barplot(
        skills_pd,
        x="Skill_Type",
        y="Frequency",
        hue="Skill_Type",
        palette="Blues_r",
        legend=False,
    )
    plt.ylabel("Frequency")
    plt.title("Histogram skills frequency")
    plt.tight_layout()
    plt.show()


def plot_skills_per_category(cv_data: pl.DataFrame, skill_type: str, top_n: int = 5) -> pl.DataFrame:
    """ """
    filtered_data = cv_data.filter(cv_data["Skill_Type"] == skill_type)
    skills_frequency = compute_skills_frequency(filtered_data, "Skill")
    skills_pd = skills_frequency.head(top_n).to_pandas()

    plt.figure(figsize=(7, 5))
    sns.barplot(
        y=skills_pd["Skill"],
        x=skills_pd["Frequency"],
        hue=skills_pd["Skill"],
        palette="Blues_r",
        legend=False,
    )
    plt.xlabel("Frequency")
    plt.ylabel("Skill")
    plt.title(f"Top {top_n} most frequent skills on: {skill_type}")
    plt.tight_layout()
    plt.show()

    return skills_frequency
