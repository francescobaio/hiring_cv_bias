"""Loading of the parser output, of the false negatives found in the CV text and of the
ESCO occupation labels, plus the deduplication of the (candidate, job) pairs."""

from typing import Literal

import polars as pl

from hiring_cv_bias.config import (
    DRIVING_LICENSE_FALSE_NEGATIVES_PATH,
    JOB_TITLE_FALSE_NEGATIVES_PATH,
    JOBS_IT_PATH,
    JOBS_PATH,
    LANGUAGE_SKILL_FALSE_NEGATIVES_PATH,
    PARSED_DATA_TRANSLATED_PATH,
)
from hiring_cv_bias.utils import load_data

FN_PATHS = {
    "driving_license": DRIVING_LICENSE_FALSE_NEGATIVES_PATH,
    "language_skill": LANGUAGE_SKILL_FALSE_NEGATIVES_PATH,
    "job_title": JOB_TITLE_FALSE_NEGATIVES_PATH,
}


def dedupe_pairs(pairs: pl.DataFrame) -> pl.DataFrame:
    """Drop repeated (candidate, job) rows, keeping the first one."""
    return pairs.unique(
        subset=["CANDIDATE_ID", "Job_id"], keep="first", maintain_order=True
    )


def load_parser_output() -> pl.DataFrame:
    """One row per (candidate, skill) with the Italian translation in ``Skill_it``."""
    return load_data(PARSED_DATA_TRANSLATED_PATH).drop("").drop_nulls("Skill")


def load_false_negatives(
    kind: Literal["driving_license", "language_skill", "job_title"],
) -> pl.DataFrame:
    """One row per (candidate, item missed by the parser)."""
    return load_data(FN_PATHS[kind]).select("CANDIDATE_ID", "skill")


def italian_job_titles(
    occupations_en: pl.DataFrame, occupations_it: pl.DataFrame
) -> dict[str, str]:
    """English ESCO title (preferred or alternative label, lower-cased) -> Italian preferred
    label of the same occupation, joined on ``conceptUri``. The Italian label is lower-cased
    and keeps the first of ESCO's gendered variants (``cassiere/cassiera`` -> ``cassiere``),
    like the ``Jobs`` vocabulary. Occupations without an Italian label are left out."""
    labels = dict(
        zip(
            occupations_it["conceptUri"].to_list(),
            occupations_it["preferredLabel"].to_list(),
            strict=True,
        )
    )
    out: dict[str, str] = {}
    rows = occupations_en.select(
        "conceptUri", "preferredLabel", "altLabels"
    ).iter_rows()
    for uri, preferred, alts in rows:
        label = labels.get(uri)
        if not label:
            continue
        for en in [preferred, *(alts or "").split("\n")]:
            en = en.strip().lower()
            # a label shared by several occupations keeps its first (preferred) meaning
            if en and en not in out:
                out[en] = label.split("/")[0].strip().lower()
    return out


def load_esco_italian_titles() -> dict[str, str]:
    """``italian_job_titles`` on ``occupations_en.csv`` and ``occupations_it.csv`` (the
    Italian preferred labels of the same ESCO occupations, downloaded once)."""
    return italian_job_titles(pl.read_csv(JOBS_PATH), pl.read_csv(JOBS_IT_PATH))
