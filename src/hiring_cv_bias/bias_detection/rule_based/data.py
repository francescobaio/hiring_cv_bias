import polars as pl
from hiring_cv_bias.config import (
    CANDIDATE_CVS_TRANSLATED_PATH,
    PARSED_DATA_PATH,
    REVERSE_MATCHING_PATH,
)
from hiring_cv_bias.utils import load_data, load_excel_sheets

from .extractors import extract_driver_license
from .utils import clean_cv


def prepare_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load CVs, parser output and join gender.
    """
    # 1) raw load
    df_cv_raw = load_data(CANDIDATE_CVS_TRANSLATED_PATH)
    df_skills = load_data(PARSED_DATA_PATH)

    # 2) get gender
    df_gender = load_excel_sheets(REVERSE_MATCHING_PATH, ["Candidates"])["Candidates"]

    # 3) join ID + Gender
    df_cv = df_cv_raw.join(
        df_gender.select(["CANDIDATE_ID", "Gender"]),
        on="CANDIDATE_ID",
        how="inner",
    )

    # 4) clean Translated_CV e add driving license flag
    df_cv = df_cv.with_columns(
        [
            pl.col("Translated_CV")
            .map_elements(clean_cv, return_dtype=str)
            .alias("cleaned_cv"),
            pl.col("Translated_CV")
            .map_elements(lambda t: bool(extract_driver_license(t)), return_dtype=bool)
            .alias("has_driving_license"),
        ]
    )

    return df_cv, df_skills
