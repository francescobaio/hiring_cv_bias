import re
from typing import Any, List, Pattern

import matplotlib.pyplot as plt
import polars as pl
import polars_ds as pds
import seaborn as sns


def inspect_missing(df: pl.DataFrame) -> pl.DataFrame:
    total = df.height

    # count nulls per column, melt into (column, n_missing)
    missing_counts = (
        df.null_count().melt().rename({"variable": "column", "value": "n_missing"})
    )

    # compute % missing per column, melt into (column, pct_missing)
    pct_missing = (
        df.null_count()
        .with_columns([(pl.col(c) / total * 100).round(2).alias(c) for c in df.columns])
        .melt()
        .rename({"variable": "column", "value": "pct_missing"})
    )

    # join counts and percentages
    stats = missing_counts.join(pct_missing, on="column")

    # print summary
    print("Missing value summary:\n", stats)
    nonzero = stats.filter(pl.col("pct_missing") > 0)
    if nonzero.height > 0:
        print("\nColumns with > 0% missing values:\n", nonzero)
    else:
        print("\nNo missing values detected in any column.")

    return stats


def filter_out_candidate_ids(
    df: pl.DataFrame,
    id_list: List[Any],
    id_col: str = "CANDIDATE_ID",
    df_name: str = "DataFrame",
    description: str = "",
) -> pl.DataFrame:
    """ """
    original = df.height
    cleaned = df.filter(~pl.col(id_col).is_in(id_list))
    remaining = cleaned.height
    dropped = original - remaining
    pct = (dropped / original * 100) if original else 0.0

    if description:
        print(f"{df_name} with {description}: {dropped} out of {original} total")
        print(f"That is {pct:.2f}% of all {df_name.lower()}")

    print(f"{df_name} -> Original rows: {original}")
    print(f"{df_name} -> Dropped rows:  {dropped} (IDs in provided list)")
    print(f"{df_name} -> Remaining rows: {remaining}")

    return cleaned


# Regex to detect placeholder runs (e.g. “XXXXX…”)
_PLACEHOLDER_PATTERN = re.compile(r"X{5,}")


def find_dropped_skill_rows(
    df: pl.DataFrame,
    skill_col: str = "Skill",
    id_col: str = "CANDIDATE_ID",
    min_len: int = 2,
    max_len: int = 100,
    placeholder_pattern: Pattern[str] = _PLACEHOLDER_PATTERN,
) -> pl.DataFrame:
    # clean mask: length, has_letter, no_placeholder
    length = pl.col(skill_col).map_elements(
        lambda s, *_: len(s or ""), return_dtype=pl.Int64
    )
    valid_length = (length >= min_len) & (length <= max_len)
    has_letter = pl.col(skill_col).map_elements(
        lambda s, *_: bool(re.search(r"[A-Za-z]", s or "")), return_dtype=pl.Boolean
    )
    no_placeholder = pl.col(skill_col).map_elements(
        lambda s, *_: not bool(placeholder_pattern.search(s or "")),
        return_dtype=pl.Boolean,
    )
    keep_mask = valid_length & has_letter & no_placeholder

    cleaned = df.filter(keep_mask)
    dropped = df.join(
        cleaned.select([id_col, skill_col, "Skill_Type"]),
        on=[id_col, skill_col, "Skill_Type"],
        how="anti",
    )
    return dropped


def cramers_v(df: pl.DataFrame, x: str, y: str) -> float:
    # chi-square
    chi2 = df.select(pds.chi2(x, y).alias("chi2")).item()["statistic"]
    n = df.height
    k = min(df.n_unique(x), df.n_unique(y))
    return (chi2 / (n * (k - 1))) ** 0.5


def plot_cramer_matrix(df: pl.DataFrame) -> None:
    df = df.drop("index")
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="Blues",
        xticklabels=df.columns,
        yticklabels=df.columns,
        square=True,
        cbar=False,
    )
    plt.title("Cramér’s V Matrix")
    plt.tight_layout()
    plt.show()
