from re import Pattern
from typing import List, Optional

import polars as pl
from IPython.display import display

from hiring_cv_bias.bias_detection.rule_based.evaluation.compare_parser import (
    error_rates_by_group,
)
from hiring_cv_bias.bias_detection.rule_based.evaluation.metrics import Result

RED, RESET = "\033[31m", "\033[0m"


def print_report(
    result: Result,
    df_population: pl.DataFrame,
    reference_col: str,
    group_col: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    disparate_impact: bool = True,
) -> None:
    print(
        f"TP: {result.conf.tp}, FP: {result.conf.fp}, TN: {result.conf.tn}, FN: {result.conf.fn}"
    )
    print(
        f"Accuracy: {result.conf.accuracy:.3f}, "
        f"Precision: {result.conf.precision:.3f}, "
        f"Recall: {result.conf.equality_of_opportunity:.3f}, "
        f"F1: {result.conf.f1:.3f}\n"
    )

    if group_col:
        df_rates = error_rates_by_group(
            result,
            df_population,
            reference_col=reference_col,
            group_col=group_col,
            metrics=metrics,
            disparate_impact=disparate_impact,
        )
        print(f"Error and rates by {group_col}:\n")
        display(df_rates)
    else:
        tot = df_population.height
        print("Overall FP-rate:", round(len(result.fp_rows) / tot, 3))
        print("Overall FN-rate:", round(len(result.fn_rows) / tot, 3))


def highlight_snippets(text: str, pattern: Pattern[str], context_chars=40) -> List[str]:
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


def print_highlighted_cv(row: dict, pattern: Pattern[str]) -> None:
    header = f"\nCANDIDATE ID: {row['CANDIDATE_ID']} - GENERE: {row['Gender']}"
    reason = f"Reason: {row['reason']}"
    separator = "-" * 80
    snippets = highlight_snippets(row["cv_text"], pattern=pattern)
    print(header)
    print(reason)
    print(separator)
    for snippet in snippets:
        print(snippet)
    print(separator)
