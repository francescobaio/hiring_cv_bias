import re

import polars as pl

from hiring_cv_bias.config import (
    CANDIDATE_CVS_TRANSLATED_PATH,
    PARSED_DATA_PATH,
    REVERSE_MATCHING_PATH,
)
from hiring_cv_bias.exploration.gender_analysis import get_category_distribution
from hiring_cv_bias.utils import load_data, load_excel_sheets

driver_license_pattern = re.compile(
    r"(?:"
    r"driver[’']?s?\s+license|"
    r"drivers?\s+license|"
    r"driving\s+license|"
    r"licensed\W+driver|"  # "licensed driver"
    r"license:\s*[a-z]{1,2}(?:\s*[-,/]\s*[a-z]{1,2})?"  # "license: am - b", "license:am/b", "license: am,b"
    r")",
    re.IGNORECASE,
)


def has_driver_license(text: str) -> bool:
    text = text.lower()
    return bool(driver_license_pattern.search(text))


def prepare_data():
    df_cv = load_data(CANDIDATE_CVS_TRANSLATED_PATH)
    df_skills = load_data(PARSED_DATA_PATH)
    dfs = load_excel_sheets(REVERSE_MATCHING_PATH, ["Candidates"])

    df_cv_with_gender = df_cv.join(
        dfs["Candidates"].select(["CANDIDATE_ID", "Gender"]),
        on="CANDIDATE_ID",
        how="inner",
    )

    df_cv_with_gender = df_cv_with_gender.with_columns(
        pl.col("Translated_CV")
        .map_elements(has_driver_license, return_dtype=bool)
        .alias("has_driving_license")
    )

    return df_cv_with_gender, df_skills


def evaluate_driver_license_extraction(
    df_cv_with_gender: pl.DataFrame, df_skills: pl.DataFrame
):
    tp = fp = tn = fn = 0
    false_positives = []
    false_negatives = []
    count = 0

    for candidate in df_cv_with_gender.iter_rows(named=True):
        candidate_id = candidate["CANDIDATE_ID"]
        candidate_gender = candidate["Gender"]
        regex_dl = candidate["has_driving_license"]

        candidate_skills_df = df_skills.filter(
            (pl.col("CANDIDATE_ID") == candidate_id)
            & (pl.col("Skill_Type") == "DRIVERSLIC")
        )
        parser_dl = candidate_skills_df.height > 0
        if parser_dl:
            count += 1

        if regex_dl and parser_dl:
            tp += 1
        elif regex_dl and not parser_dl:
            fp += 1
            false_positives.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_gender": candidate_gender,
                    "cv_text": candidate["Translated_CV"],
                    "reason": "Regex sees license, parser does NOT",
                }
            )
        elif not regex_dl and not parser_dl:
            tn += 1
        elif not regex_dl and parser_dl:
            fn += 1
            false_negatives.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_gender": candidate_gender,
                    "cv_text": candidate["Translated_CV"],
                    "reason": "Parser sees license, regex does NOT",
                }
            )

    return tp, fp, tn, fn, false_positives, false_negatives, count


def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return accuracy, precision, recall, f1


def analyze_bias_by_gender(df_cv_with_gender, false_positives, false_negatives):
    df_fp = pl.DataFrame(false_positives)
    df_fn = pl.DataFrame(false_negatives)

    fp_counts = df_fp.group_by("candidate_gender").agg(
        pl.len().alias("num_false_positives")
    )
    fn_counts = df_fn.group_by("candidate_gender").agg(
        pl.len().alias("num_false_negatives")
    )

    gender_counts_df = get_category_distribution(df_cv_with_gender, "Gender")
    gender_counts_renamed = gender_counts_df.rename(
        {"Gender": "candidate_gender", "count": "total_candidates"}
    )

    fp_joined = fp_counts.join(gender_counts_renamed, on="candidate_gender", how="left")
    fp_joined = fp_joined.with_columns(
        (pl.col("num_false_positives") / pl.col("total_candidates")).alias("fp_rate")
    )

    fn_joined = fn_counts.join(gender_counts_renamed, on="candidate_gender", how="left")
    fn_joined = fn_joined.with_columns(
        (pl.col("num_false_negatives") / pl.col("total_candidates")).alias("fn_rate")
    )

    return fp_joined, fn_joined
