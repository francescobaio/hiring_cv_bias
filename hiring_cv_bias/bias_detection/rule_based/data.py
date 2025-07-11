import polars as pl

from .extractors import extract_driver_license


def add_demographic_info(
    cv_df: pl.DataFrame,
    candidates_df: pl.DataFrame,
) -> pl.DataFrame:
    enriched_cv_df = cv_df.join(
        candidates_df.select(["CANDIDATE_ID", "Gender", "Location", "Age_bucket"]),
        on="CANDIDATE_ID",
        how="inner",
    ).with_columns(
        pl.col("Translated_CV")
        .map_elements(
            lambda t: bool(extract_driver_license(t)), return_dtype=pl.Boolean
        )
        .alias("has_driving_license")
    )

    return enriched_cv_df


def filter_unknown_and_other_rows(
    cv_df: pl.DataFrame,
) -> pl.DataFrame:
    cv_df = cv_df.filter(pl.col("Gender").is_in(["Male", "Female"]))
    cv_df = cv_df.filter(pl.col("Age_bucket").is_in(["25-34", "45-54", "55-74"]))
    return cv_df
