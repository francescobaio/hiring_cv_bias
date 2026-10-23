"""Loading of the parser output and of the false negatives found in the CV text,
plus the deduplication of the (candidate, job) pairs."""

from typing import Literal

import polars as pl

from hiring_cv_bias.config import (
    DRIVING_LICENSE_FALSE_NEGATIVES_PATH,
    JOB_TITLE_FALSE_NEGATIVES_PATH,
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
