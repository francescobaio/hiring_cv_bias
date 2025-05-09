import re
import unicodedata as ud
from typing import Iterable, List

from hiring_cv_bias.bias_detection.rule_based.evaluation.bias import (
    error_rates_by_group,
)
from hiring_cv_bias.bias_detection.rule_based.evaluation.metrics import Conf, scores

_CV_HEADER = re.compile(r"cv anonimizzato\s*:", re.I)
_WS = re.compile(r"\s+")

RED, RESET = "\033[31m", "\033[0m"


def clean_cv(text: str) -> str:
    """
    Normalise a CV string:
    * remove the anonymised header
    * collapse line breaks / multiple spaces
    * Unicode-normalise (NFKC)
    """
    text = _CV_HEADER.sub("", text)
    text = text.replace('"""', "")
    text = _WS.sub(" ", text)
    return ud.normalize("NFKC", text).strip()


def print_report(
    conf: Conf,
    df_population,  # DataFrame con la colonna Gender
    fp_rows: Iterable[dict],
    fn_rows: Iterable[dict],
):
    """Pretty-print confusion-matrix scores and group error rates."""
    s = scores(conf)
    print(f"TP: {conf.tp}, FP: {conf.fp}, TN: {conf.tn}, FN: {conf.fn}")
    print(
        f"Accuracy: {s['accuracy']:.3f}, "
        f"Precision: {s['precision']:.3f}, "
        f"Recall: {s['recall']:.3f}, "
        f"F1: {s['f1']:.3f}\n"
    )

    fp_rate, fn_rate = error_rates_by_group(df_population, fp_rows, fn_rows)
    print("False-positive rate by gender\n", fp_rate)
    print("False-negative rate by gender\n", fn_rate)


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
