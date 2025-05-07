from typing import Callable, List, Tuple

import polars as pl
from hiring_cv_bias.bias_detection.rule_based.config import (
    DRIVER_LICENSE_SNIPPET_PATTERN,
    LANGUAGE_REGEXES,
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
from hiring_cv_bias.utils import load_data, load_excel_sheets


def has_driver_license(text: str) -> bool:
    text = text.lower()
    return bool(driver_license_pattern_it.search(text))


def extract_languages_rule_based(text: str) -> List[str]:
    text = text.lower()
    return [lang for lang, regex in LANGUAGE_REGEXES.items() if regex.search(text)]


def extract_parser_snippets(text: str, context_chars=150) -> List[str]:
    snippets = []
    for match in DRIVER_LICENSE_SNIPPET_PATTERN.finditer(text):
        start, end = match.span()
        snippet_start = max(start - context_chars, 0)
        snippet_end = min(end + context_chars, len(text))
        before = text[snippet_start:start]
        matched = match.group(0)
        after = text[end:snippet_end]
        snippets.append(f"... {before}...{matched}...{after} ...")
    return snippets


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


def evaluate_rule_based_extraction(
    df_cv_with_gender: pl.DataFrame,
    df_skills: pl.DataFrame,
    skill_type: str,
    rule_based_extractor: Callable[[str], List[str]],
    normalize_parser_skill: Callable[[str], str] = lambda x: x.lower(),
) -> Tuple[int, int, int, int, list, list, int]:
    tp = fp = tn = fn = 0
    false_positives = []
    false_negatives = []
    count = 0

    for candidate in df_cv_with_gender.iter_rows(named=True):
        candidate_id = candidate["CANDIDATE_ID"]
        candidate_gender = candidate["Gender"]
        cv_text = candidate["CV_text_anon"]
        cleaned_text = clean_cv(cv_text)

        rule_skills = set(rule_based_extractor(cleaned_text))

        candidate_skills_df = df_skills.filter(
            (pl.col("CANDIDATE_ID") == candidate_id)
            & (pl.col("Skill_Type") == skill_type)
        )
        parser_skills = set(
            normalize_parser_skill(skill)
            for skill in candidate_skills_df["Skill"].to_list()
        )

        matched_skills = rule_skills & parser_skills
        unmatched_regex = rule_skills - parser_skills
        unmatched_parser = parser_skills - rule_skills

        tp += len(matched_skills)
        fp += len(unmatched_regex)
        fn += len(unmatched_parser)

        if not rule_skills and not parser_skills:
            tn += 1

        if rule_skills or parser_skills:
            count += 1

        for skill in unmatched_regex:
            false_positives.append(
                {
                    "CANDIDATE_ID": candidate_id,
                    "candidate_gender": candidate_gender,
                    "cv_text": cv_text,
                    "skill": skill,
                    "reason": "Regex sees skill, parser does NOT",
                }
            )

        for skill in unmatched_parser:
            false_negatives.append(
                {
                    "CANDIDATE_ID": candidate_id,
                    "candidate_gender": candidate_gender,
                    "cv_text": cv_text,
                    "skill": skill,
                    "reason": "Parser sees skill, regex does NOT",
                }
            )

    return tp, fp, tn, fn, false_positives, false_negatives, count


def highlight_snippets(text: str, pattern, context_chars=75) -> List[str]:
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
    return snippets or ["No occurrence found."]


def print_highlighted_cv(row: dict, pattern) -> None:
    header = (
        f"\nCANDIDATE ID: {row['CANDIDATE_ID']} - GENERE: {row['candidate_gender']}"
    )
    reason = f"Motivo: {row['reason']}"
    separator = "-" * 80
    snippets = highlight_snippets(row["cv_text"], pattern=pattern)
    print(header)
    print(reason)
    print(separator)
    for snippet in snippets:
        print(snippet)
    print(separator)


def export_false_negatives_for_manual_labelling(
    df_cv: pl.DataFrame,
    df_skills: pl.DataFrame,
    skill_type: str,
    rule_based_extractor: Callable[[str], List[str]],
    output_path: str = "false_negatives_manual_labelling.csv",
    sample_size: int = 50,
):
    parser_ids = df_skills.filter(pl.col("Skill_Type") == skill_type).unique(
        "CANDIDATE_ID"
    )

    df_parser_yes = df_cv.join(parser_ids, on="CANDIDATE_ID", how="inner")
    regex_labels = [
        len(rule_based_extractor(text)) > 0 for text in df_parser_yes["CV_text_anon"]
    ]

    mask = [not r for r in regex_labels]
    df_filtered = df_parser_yes.filter(pl.Series("", mask))

    df_to_label = df_filtered.with_columns(
        [
            pl.Series("regex_label", [0] * len(df_filtered)),
            pl.Series("parser_label", [1] * len(df_filtered)),
            pl.Series("manual_label", [""] * len(df_filtered)),
            df_filtered["CV_text_anon"].alias("cv_text"),
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
            "cv_text",
        ]
    ).write_csv(output_path, separator=";", quote_style="always")

    print(f"CSV false negatives exported with {df_to_label.height} rows: {output_path}")
