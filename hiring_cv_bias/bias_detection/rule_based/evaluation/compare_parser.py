from collections import namedtuple
from typing import Any, Callable, Dict, Set

import polars as pl
from tqdm.notebook import tqdm

Conf = namedtuple("Conf", "tp fp tn fn")
Result = namedtuple("Result", "conf fp_rows fn_rows")


def compare(
    df_cv: pl.DataFrame,
    df_parser: pl.DataFrame,
    skill_type: str,
    extractor: Callable[[str], Set[str]],
    norm: Callable[[str], str] = str.lower,
) -> Result:
    """
    Compare rule-based extraction (truth) with parser output.
    """
    tp = fp = tn = fn = 0
    fp_rows, fn_rows = [], []

    for row in tqdm(df_cv.iter_rows(named=True), total=df_cv.height):
        cid, gender, raw = row["CANDIDATE_ID"], row["Gender"], row["cleaned_cv"]

        truth = extractor(raw)  # rule‑based “ground‑truth”

        # set comprehension give or "driver_license" or None as truth
        parser = {
            norm(s)
            for s in df_parser.filter(
                (pl.col("CANDIDATE_ID") == cid) & (pl.col("Skill_Type") == skill_type)
            )["Skill"].to_list()
            if isinstance(s, str)
        }

        tp += len(truth & parser)
        fn += len(truth - parser)  # missed by parser
        fp += len(parser - truth)  # hallucinated by parser
        if not truth and not parser:
            tn += 1

        fp_rows += [
            {
                "CANDIDATE_ID": cid,
                "Gender": gender,
                "skill": s,
                "cv_text": raw,
                "cv_italian": row["CV_text_anon"],
                "reason": "Parser output contains skill not found by rule-based extractor. ",
            }
            for s in parser - truth
        ]
        fn_rows += [
            {
                "CANDIDATE_ID": cid,
                "Gender": gender,
                "skill": s,
                "cv_text": raw,
                "cv_italian": row["CV_text_anon"],
                "reason": "Rule-based extractor found skill but parser missed it.",
            }
            for s in truth - parser
        ]

    return Result(Conf(tp, fp, tn, fn), fp_rows, fn_rows)


def compute_candidate_coverage(
    df_cv: pl.DataFrame,
    df_sk: pl.DataFrame,
    skill_type: str,
    extractor: Callable[[str], Set[str]],
    text_col: str = "cleaned_cv",
    id_col: str = "CANDIDATE_ID",
) -> Dict[str, Any]:
    """
    Compute coverage stats for one skill category.
    """
    # 1) find all candidates whose cleaned text triggers the extractor
    mask_regex = df_cv[text_col].map_elements(
        lambda t: bool(extractor(t)), return_dtype=bool
    )
    regex_series = df_cv.filter(mask_regex)[id_col]
    regex_ids = set(regex_series.unique().to_list())

    # 2) parser side
    df_sk_skill = df_sk.filter(pl.col("Skill_Type") == skill_type)
    parser_series = df_sk_skill[id_col]
    parser_ids = set(parser_series.unique().to_list())
    num_parser_occurrences = df_sk_skill.height

    # 3) set operations
    common_ids = regex_ids & parser_ids
    only_regex_ids = regex_ids - parser_ids
    only_parser_ids = parser_ids - regex_ids

    # 4) counts
    stats: Dict[str, Any] = {
        "regex_ids": regex_ids,
        "parser_ids": parser_ids,
        "common_ids": common_ids,
        "only_regex_ids": only_regex_ids,
        "only_parser_ids": only_parser_ids,
        "num_regex_candidates": len(regex_ids),
        "num_parser_unique": len(parser_ids),
        "num_parser_occurrences": num_parser_occurrences,
        "num_common_candidates": len(common_ids),
        "num_only_regex_candidates": len(only_regex_ids),
        "num_only_parser_candidates": len(only_parser_ids),
    }
    return stats


def print_candidate_coverage(
    df_cv: pl.DataFrame,
    df_sk: pl.DataFrame,
    skill_type: str,
    extractor: Callable[[str], Set[str]],
    text_col: str = "cleaned_cv",
    id_col: str = "CANDIDATE_ID",
) -> Dict[str, Any]:
    """
    Compute and print the main coverage numbers for a skill.
    """
    stats = compute_candidate_coverage(
        df_cv, df_sk, skill_type, extractor, text_col, id_col
    )

    print(f"Regex-positive candidates        : {stats['num_regex_candidates']}")
    print(f"Parser-positive unique candidates: {stats['num_parser_unique']}")
    print(f"Parser total occurrences         : {stats['num_parser_occurrences']}\n")

    print(f"- Both regex & parser   : {stats['num_common_candidates']}")
    print(f"- Only regex            : {stats['num_only_regex_candidates']}")
    print(f"- Only parser           : {stats['num_only_parser_candidates']}\n")

    return stats
