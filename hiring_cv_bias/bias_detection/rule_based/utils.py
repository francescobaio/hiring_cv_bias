from typing import Dict, List, Optional

from hiring_cv_bias.bias_detection.rule_based.evaluation.bias import (
    error_rates_by_group,
)
from hiring_cv_bias.bias_detection.rule_based.evaluation.metrics import Conf, scores

RED, RESET = "\033[31m", "\033[0m"


def print_report(
    conf: Conf,
    df_population,
    fp_rows: List[Dict],
    fn_rows: List[Dict],
    group_col: Optional[str] = None,
):
    s = scores(conf)
    print(f"TP: {conf.tp}, FP: {conf.fp}, TN: {conf.tn}, FN: {conf.fn}")
    print(
        f"Accuracy: {s['accuracy']:.3f}, "
        f"Precision: {s['precision']:.3f}, "
        f"Recall: {s['recall']:.3f}, "
        f"F1: {s['f1']:.3f}\n"
    )

    if group_col:
        fp_rate, fn_rate = error_rates_by_group(
            df_population, fp_rows, fn_rows, group_col=group_col
        )
        print(f"False positive rate by {group_col}:\n", fp_rate)
        print(f"False negative rate by {group_col}:\n", fn_rate)
    else:
        total = df_population.height
        num_fp = len(fp_rows)
        num_fn = len(fn_rows)

        fp_rate = num_fp / total if total > 0 else 0.0
        fn_rate = num_fn / total if total > 0 else 0.0

        print("Overall false positive rate:", round(fp_rate, 3))
        print("Overall false negative rate:", round(fn_rate, 3))


def highlight_snippets(text: str, pattern, context_chars=100) -> List[str]:
    snippets = []
    for match in pattern.finditer(text):
        start, end = match.span()
        snippet_start = max(start - context_chars, 0)
        snippet_end = min(end + context_chars, len(text))
        before = text[snippet_start:start]
        match_text = text[start:end]
        after = text[end:snippet_end]

        colored_match_text = f"{RED}{match_text}{RESET}"
        snippets.append(f"{before}{colored_match_text}{after}")
    return snippets or ["No occurrence found."]


def print_highlighted_cv(row: dict, pattern) -> None:
    header = f"\nCANDIDATE ID: {row['CANDIDATE_ID']} - GENERE: {row['Gender']}"
    reason = f"Motivo: {row['reason']}"
    separator = "-" * 80
    snippets = highlight_snippets(row["cv_text"], pattern=pattern)
    print(header)
    print(reason)
    print(separator)
    for snippet in snippets:
        print(snippet)
    print(separator)
