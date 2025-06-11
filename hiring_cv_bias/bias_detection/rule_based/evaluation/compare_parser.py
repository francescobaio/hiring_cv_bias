from collections import namedtuple
from typing import Callable, Set

import polars as pl
from tqdm.notebook import tqdm

Conf = namedtuple("Conf", "tp fp tn fn")
Result = namedtuple("Result", "conf fp_rows fn_rows")


def compute_candidate_coverage(
    df_cv: pl.DataFrame,
    df_parser: pl.DataFrame,
    skill_type: str,
    extractor: Callable[[str], Set[str]],
    norm: Callable[[str], str] = str.lower,
    verbose: bool = True,
) -> Result:
    """
    Compare rule-based extraction (truth) with parser output.
    """
    tp = fp = tn = fn = 0
    fp_rows, fn_rows = [], []
    truth_pos_ids: set[str] = set()
    parser_pos_ids: set[str] = set()
    parser_occurrences = 0

    for row in tqdm(df_cv.iter_rows(named=True), total=df_cv.height):
        cid, gender, raw = row["CANDIDATE_ID"], row["Gender"], row["Translated_CV"]

        truth = extractor(raw)
        if truth:
            truth_pos_ids.add(cid)

        # set comprehension give or "driver_license" or None as truth
        parser = {
            norm(s)
            for s in df_parser.filter(
                (pl.col("CANDIDATE_ID") == cid) & (pl.col("Skill_Type") == skill_type)
            )["Skill"].to_list()
            if isinstance(s, str)
        }
        if parser:
            parser_pos_ids.add(cid)
            parser_occurrences += len(parser)

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
                "reason": "Parser output contains skill not found by rule‐based extractor.",
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
                "reason": "Rule‐based extractor found skill but parser missed it.",
            }
            for s in truth - parser
        ]

    if verbose:
        both_ids = truth_pos_ids & parser_pos_ids
        only_truth = truth_pos_ids - parser_pos_ids
        only_parser = parser_pos_ids - truth_pos_ids

        print(f"Regex positive candidates        : {len(truth_pos_ids)}")
        print(f"Parser positive unique candidates: {len(parser_pos_ids)}")

        print(f"- Both regex & parser   : {len(both_ids)}")
        print(f"- Only regex            : {len(only_truth)}")
        print(f"- Only parser           : {len(only_parser)}\n")

    return Result(Conf(tp, fp, tn, fn), fp_rows, fn_rows)
