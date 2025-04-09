from typing import List

import polars as pl
from hiring_cv_bias.bias_detection.rule_based.config import (
    RED,
    RESET,
    driver_license_pattern_it,
)
from hiring_cv_bias.bias_detection.rule_based.utils import clean_cv
from hiring_cv_bias.config import (
    CANDIDATE_CVS_TRANSLATED_PATH,
    PARSED_DATA_PATH,
    REVERSE_MATCHING_PATH,
)
from hiring_cv_bias.exploration.gender_analysis import get_category_distribution
from hiring_cv_bias.utils import load_data, load_excel_sheets


def has_driver_license(text: str) -> bool:
    text = text.lower()
    return bool(driver_license_pattern_it.search(text))


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
        [
            pl.col("CV_text_anon")
            .map_elements(clean_cv, return_dtype=str)
            .alias("cleaned_cv"),
            pl.col("CV_text_anon")
            .map_elements(lambda t: has_driver_license(clean_cv(t)), return_dtype=bool)
            .alias("has_driving_license"),
        ]
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
                    "cv_text": candidate["CV_text_anon"],
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
                    "cv_text": candidate["CV_text_anon"],
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


def highlight_snippets(
    text: str, pattern=driver_license_pattern_it, context_chars=75
) -> List[str]:
    snippets = []
    for match in pattern.finditer(text):
        start, end = match.span()
        snippet_start = max(start - context_chars, 0)
        snippet_end = min(end + context_chars, len(text))
        before = text[snippet_start:start]
        match_text = text[start:end]
        after = text[end:snippet_end]

        colored_match_text = f"{RED}{match_text}{RESET}"

        snippets.append(f"... {before}...{colored_match_text}...{after} ...")
    return snippets or ["No occurrence founded."]


def print_highlighted_cv(row: dict):
    header = (
        f"\nCANDIDATE ID: {row['candidate_id']} - GENERE: {row['candidate_gender']}"
    )
    reason = f"Motivo: {row['reason']}"
    separator = "-" * 80
    snippets = highlight_snippets(row["cv_text"])
    print(header)
    print(reason)
    print(separator)
    for snippet in snippets:
        print(snippet)
    print(separator)


def export_false_negatives_for_manual_labelling(
    df_cv: pl.DataFrame,
    df_skills: pl.DataFrame,
    output_path: str = "false_negatives_manual_labelling.csv",
    sample_size: int = 50,
):
    parser_ids = df_skills.filter(pl.col("Skill_Type") == "DRIVERSLIC").unique(
        "CANDIDATE_ID"
    )

    df_parser_yes = df_cv.join(parser_ids, on="CANDIDATE_ID", how="inner")
    regex_labels = [has_driver_license(text) for text in df_parser_yes["CV_text_anon"]]

    mask = [not r for r in regex_labels]
    df_filtered = df_parser_yes.filter(pl.Series("", mask))
    print(df_filtered)

    df_to_label = df_filtered.with_columns(
        [
            pl.Series("regex_label", [0] * len(df_filtered)),
            pl.Series("parser_label", [1] * len(df_filtered)),
            pl.Series("manual_label", [""] * len(df_filtered)),
            pl.col("CV_text_anon").alias("full_cv_text"),
        ]
    )

    if sample_size < df_to_label.height:
        df_to_label = df_to_label.sample(n=sample_size, seed=42)

    df_to_label.select(
        [
            "CANDIDATE_ID",
            "regex_label",
            "parser_label",
            "manual_label",
            "full_cv_text",
        ]
    ).write_csv(output_path, separator=";", quote_style="always")

    print(f"CSV false negatives exported with {df_to_label.height} rows: {output_path}")
