from typing import Any, Dict, List, Set

import polars as pl

from hiring_cv_bias.bias_detection.rule_based.evaluation.metrics import Conf, Result


def extract_skill_cases(
    res_job: Result,
    df_cv: pl.DataFrame,
    skill_variants: Set[str],
):
    candidate_ids = set(df_cv["CANDIDATE_ID"].to_list())
    fp_skills: List[Dict[str, Any]] = []
    tp_skills: List[Dict[str, Any]] = []
    fn_skills: List[Dict[str, Any]] = []

    def _collect_skills(rows, skills):
        for row in rows:
            if row["skill"] in skill_variants:
                skills.append(row)
                candidate_ids.discard(row["CANDIDATE_ID"])

    _collect_skills(res_job.fp_rows, fp_skills)
    _collect_skills(res_job.tp_rows, tp_skills)
    _collect_skills(res_job.fn_rows, fn_skills)

    tn_skills = []
    for cid in candidate_ids:
        row = df_cv.filter(pl.col("CANDIDATE_ID") == cid)
        tn_skills.append(
            {
                "CANDIDATE_ID": cid,
                "Gender": row["Gender"].item(),
                "Location": row["Location"].item(),
                "length": row["length"].item(),
                "skill": None,
                "cv_text": row["Translated_CV"].item(),
                "cv_italian": row["CV_text_anon"].item(),
                "reason": "No skill found by either extractor or parser.",
            }
        )

    conf = Conf(
        tp=len(tp_skills), fp=len(fp_skills), tn=len(tn_skills), fn=len(fn_skills)
    )

    return Result(conf, tp_skills, fp_skills, fn_skills, tn_skills)
