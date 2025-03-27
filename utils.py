import polars as pl
from config import CANDIDATE_CVS_PATH, PARSED_DATA_PATH, REVERSE_MATCHING_PATH
from sklearn.preprocessing import LabelEncoder
from typing import Any


def load_data():
    df_skills = pl.read_csv(
        PARSED_DATA_PATH,
        separator=";",
    )
    df_info_matching = pl.read_excel(
        REVERSE_MATCHING_PATH,
        sheet_name="ReverseMatching",
    )
    df_info_candidates = pl.read_excel(
        REVERSE_MATCHING_PATH,
        sheet_name="Candidates",
    )
    df_info_jobs = pl.read_excel(
        REVERSE_MATCHING_PATH,
        sheet_name="Jobs",
    )

    # df_info = df_info.with_columns(
    #     df_info.select(
    #         pl.col("cand_id")
    #         .str.replace_all(",", "", literal=True)
    #         .cast(pl.Int64)
    #         .alias("cand_id")
    #     )
    # )
    # df_cv = pl.read_csv(
    #     "./data/Adecco_csv_parsed_data/cvs_anon.csv", separator=";"
    # ).rename({"CANDIDATE_ID": "cand_id"})

    return df_skills, df_info_matching, df_info_candidates, df_info_jobs


def localize(latitude: float) -> str:
    if latitude > 44.5:
        return "NORTH"
    if latitude < 40:
        return "SOUTH"
    return "CENTER"


def to_categorical(column: pl.Series) -> pl.Series:
    encoder = LabelEncoder()
    encoded_column = pl.from_numpy(encoder.fit_transform(column)).to_series()
    return encoded_column


def create_mask(column: pl.Series, value: Any) -> pl.Series:
    mask_df = column.to_frame().with_columns(
        pl.when(pl.col(column.name) == value)
        .then(1)
        .otherwise(0)
        .alias(column.name + "_mask")
    )
    return mask_df.get_column(column.name + "_mask")
