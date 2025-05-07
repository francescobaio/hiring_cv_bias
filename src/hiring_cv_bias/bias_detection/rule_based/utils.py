import re

import polars as pl
from hiring_cv_bias.exploration.gender_analysis import get_category_distribution


def clean_cv(cv: str) -> str:
    cv = cv.replace("CV anonimizzato:", "")
    cv = cv.replace('"""', "")
    cv = re.sub(r"[\n\r]+", " ", cv)
    cv = re.sub(r"\s{2,}", " ", cv)
    return cv.strip()


def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return accuracy, precision, recall, f1


def print_report(tp, fp, tn, fn, df_cv_with_gender, fp_list, fn_list):
    accuracy, precision, recall, f1 = compute_metrics(tp, fp, tn, fn)
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n"
    )

    fp_rates, fn_rates = analyze_bias_by_gender(df_cv_with_gender, fp_list, fn_list)
    print("False Positives:\n", fp_rates)
    print("False Negatives:\n", fn_rates)


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
