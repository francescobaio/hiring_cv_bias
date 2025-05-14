from typing import List

import polars as pl


def inspect_missing(df: pl.DataFrame) -> pl.DataFrame:
    """ """
    total = df.height

    # 1) Count nulls per column, melt into (column, n_missing)
    missing_counts = (
        df.null_count().melt().rename({"variable": "column", "value": "n_missing"})
    )

    # 2) Compute % missing per column, melt into (column, pct_missing)
    pct_missing = (
        df.null_count()
        .with_columns([(pl.col(c) / total * 100).round(2).alias(c) for c in df.columns])
        .melt()
        .rename({"variable": "column", "value": "pct_missing"})
    )

    # 3) Join counts and percentages
    stats = missing_counts.join(pct_missing, on="column")

    # 4) Print summary
    print("Missing value summary:\n", stats)
    nonzero = stats.filter(pl.col("pct_missing") > 0)
    if nonzero.height > 0:
        print("\nColumns with > 0% missing values:\n", nonzero)
    else:
        print("\nNo missing values detected in any column.")

    return stats


def filter_out_candidate_ids(
    df: pl.DataFrame,
    id_list: List[int],
    id_col: str = "CANDIDATE_ID",
    df_name: str = "DataFrame",
) -> pl.DataFrame:
    """ """
    original = df.height
    cleaned = df.filter(~pl.col(id_col).is_in(id_list))
    remaining = cleaned.height
    dropped = original - remaining

    print(f"{df_name} -> Original rows: {original}")
    print(f"{df_name} -> Dropped rows:  {dropped} (IDs in provided list)")
    print(f"{df_name} -> Remaining rows: {remaining}")

    return cleaned
