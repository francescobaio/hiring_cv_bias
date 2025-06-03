import polars as pl

from hiring_cv_bias.exploration.gender_analysis import get_category_distribution


def error_rates_by_group(
    df_population: pl.DataFrame,
    fp_rows: list[dict],
    fn_rows: list[dict],
    group_col: str = "Gender",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Compute FP-rate and FN-rate per demographic group.
    """
    df_fp = pl.DataFrame(fp_rows)
    df_fn = pl.DataFrame(fn_rows)

    fp_counts = df_fp.group_by(group_col).len().rename({"len": "num_fp"})
    fn_counts = df_fn.group_by(group_col).len().rename({"len": "num_fn"})

    pop_counts = get_category_distribution(df_population, group_col).rename(
        {group_col: group_col, "count": "total"}
    )

    fp_rate = fp_counts.join(pop_counts, on=group_col, how="left").with_columns(
        (pl.col("num_fp") / pl.col("total")).alias("fp_rate")
    )

    fn_rate = fn_counts.join(pop_counts, on=group_col, how="left").with_columns(
        (pl.col("num_fn") / pl.col("total")).alias("fn_rate")
    )

    return fp_rate, fn_rate
