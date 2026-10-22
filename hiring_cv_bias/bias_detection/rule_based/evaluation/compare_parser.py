from collections.abc import Callable
from typing import Any

import polars as pl
from tqdm.notebook import tqdm

from hiring_cv_bias.bias_detection.rule_based.evaluation.metrics import Conf, Result


def collect_confusion_rows(
    rows: list[dict[str, Any]],
    base_row: dict[str, Any],
    keys: list[str],
    skills: set[str | None],
    reason: str,
) -> None:
    base = {k: base_row[k] for k in keys}
    for s in skills:
        entry = dict(base)
        entry["skill"] = s
        entry["cv_text"] = base_row["Translated_CV"]
        entry["cv_italian"] = base_row["CV_text_anon"]
        entry["reason"] = reason
        rows.append(entry)


def compute_candidate_coverage(
    df_cv: pl.DataFrame,
    df_parser: pl.DataFrame,
    skill_type: str,
    extractor: Callable[[str], set[str]],
    norm: Callable[[str], str] = str.lower,
    matcher: Callable[[set[str], set[str]], set[str]] | None = None,
    verbose: bool = True,
) -> Result:
    tp = fp = tn = fn = 0
    tp_rows: list[dict] = []
    fp_rows: list[dict] = []
    fn_rows: list[dict] = []
    tn_rows: list[dict] = []
    truth_ids, parser_ids = set(), set()

    features = ["CANDIDATE_ID", "Gender", "Location", "length"]

    for row in tqdm(df_cv.iter_rows(named=True), total=df_cv.height):
        cid, raw = row["CANDIDATE_ID"], row["Translated_CV"]

        truth = extractor(raw)
        if truth:
            truth_ids.add(cid)

        parser = {
            norm(s)
            for s in df_parser.filter(
                (pl.col("CANDIDATE_ID") == cid) & (pl.col("Skill_Type") == skill_type)
            )["Skill"].to_list()
            if isinstance(s, str)
        }
        if parser:
            parser_ids.add(cid)

        if matcher is not None:
            truth = matcher(truth, parser)

        # TP
        tp_skills = truth & parser
        tp += len(tp_skills)
        collect_confusion_rows(
            tp_rows,
            row,
            features,
            set(tp_skills),
            "Both regex & parser found this skill.",
        )

        # FN
        fn_skills = truth - parser
        fn += len(fn_skills)
        collect_confusion_rows(
            fn_rows,
            row,
            features,
            set(fn_skills),
            "Rule-based extractor found skill but parser missed it.",
        )

        # FP
        fp_skills = parser - truth
        fp += len(fp_skills)
        collect_confusion_rows(
            fp_rows,
            row,
            features,
            set(fp_skills),
            "Parser output contains skill not found by rule-based extractor.",
        )

        # TN
        if not truth and not parser:
            tn += 1
            collect_confusion_rows(
                tn_rows,
                row,
                features,
                {None},
                "No skill found by either extractor or parser.",
            )

    if verbose:
        both = truth_ids & parser_ids
        only_t = truth_ids - parser_ids
        only_p = parser_ids - truth_ids
        print(f"Regex positive candidates        : {len(truth_ids)}")
        print(f"Parser positive candidates: {len(parser_ids)}")
        print(f"- Both regex & parser   : {len(both)}")
        print(f"- Only regex            : {len(only_t)}")
        print(f"- Only parser           : {len(only_p)}\n")

    return Result(
        Conf(tp, fp, tn, fn),
        tp_rows,
        fp_rows,
        fn_rows,
        tn_rows,
    )


def error_rates_by_group(
    result: Result,
    df_population: pl.DataFrame,
    reference_col: str,
    group_col: str = "Gender",
    metrics: list[str] | None = None,
    disparate_impact: bool = True,
) -> pl.DataFrame:
    metrics = metrics or []
    df_tp = pl.DataFrame(result.tp_rows)
    df_fp = pl.DataFrame(result.fp_rows)
    df_fn = pl.DataFrame(result.fn_rows)
    df_tn = pl.DataFrame(result.tn_rows)

    tp_cnt = df_tp.group_by(group_col).len().rename({"len": "tp"})
    fp_cnt = df_fp.group_by(group_col).len().rename({"len": "fp"})
    fn_cnt = df_fn.group_by(group_col).len().rename({"len": "fn"})
    tn_cnt = df_tn.group_by(group_col).len().rename({"len": "tn"})

    pop_cnt = df_population.group_by(group_col).agg(pl.count().alias("total"))

    df = (
        pop_cnt.join(tp_cnt, on=group_col, how="left")
        .join(fp_cnt, on=group_col, how="left")
        .join(fn_cnt, on=group_col, how="left")
        .join(tn_cnt, on=group_col, how="left")
        .fill_null(0)
        .with_columns(pl.sum_horizontal("tp", "fp", "fn", "tn").alias("total_skills"))
    )

    df = df.with_columns(
        [
            (pl.col("fp") / pl.col("total_skills")).alias("fp_rate"),
            (pl.col("fn") / pl.col("total_skills")).alias("fn_rate"),
        ]
    )

    for m in metrics:
        if not hasattr(Conf, m):
            raise ValueError(f"Metric '{m}' not defined in Conf")
        df = df.with_columns(
            pl.struct(["tp", "fp", "tn", "fn"])
            .map_elements(
                lambda d, m=m: float(getattr(Conf(**d), m)), return_dtype=pl.Float64
            )
            .alias(m)
        )

    # compute disparate impact between groups
    if disparate_impact:
        confs = {
            row[group_col]: Conf(tp=row["tp"], fp=row["fp"], tn=row["tn"], fn=row["fn"])
            for row in df.iter_rows(named=True)
        }

        dsp_imp = [confs[i].disparate_impact(confs[reference_col]) for i in confs]
        df.insert_column(len(df.columns), pl.Series("disparate_impact", dsp_imp))

    return df
