"""Rule-based components of the score, recomputed from a profile and the job requirements:
``Card_match``, ``Languages_match`` and ``Regulated_Trainings_match`` are 1 when the candidate
has every required item (or nothing is required), ``match_skills`` is the share of required
skills owned, ``Distance_match`` is 1 within 10 km, 0.6 within 20 km, 0.2 within 25 km."""

import polars as pl

from hiring_cv_bias.reverse_matching.profiles import normalize_item, split_items
from hiring_cv_bias.reverse_matching.scoring import RULE_COMPONENTS

# Jobs sheet column -> requirement list
JOB_REQUIREMENTS = {
    "drivinglicense": "req_licenses",
    "language": "req_languages",
    "regulatedtraining": "req_regulated",
    "matterknowledge": "req_skills",
}


def job_requirements(jobs: pl.DataFrame) -> pl.DataFrame:
    return jobs.select(
        "Job_id",
        normalize_item(pl.col("professional_category")).alias("job_category"),
        *[
            split_items(pl.col(col)).alias(alias)
            for col, alias in JOB_REQUIREMENTS.items()
        ],
    )


def _subset(required: str, owned: str) -> pl.Expr:
    return (pl.col(required).list.set_difference(pl.col(owned)).list.len() == 0).cast(
        pl.Float64
    )


def _fraction(required: str, owned: str) -> pl.Expr:
    n_req = pl.col(required).list.len()
    n_hit = pl.col(required).list.set_intersection(pl.col(owned)).list.len()
    return pl.when(n_req == 0).then(0.0).otherwise(n_hit / n_req).cast(pl.Float64)


def _distance(km: pl.Expr) -> pl.Expr:
    return (
        pl.when(km <= 10)
        .then(1.0)
        .when(km <= 20)
        .then(0.6)
        .when(km <= 25)
        .then(0.2)
        .otherwise(0.0)
        .cast(pl.Float64)
    )


def add_components(
    pairs: pl.DataFrame, profile: pl.DataFrame, jobs_req: pl.DataFrame, suffix: str
) -> pl.DataFrame:
    """Add the rule-based components (named with ``suffix``, e.g. ``Card_match_O``) to ``pairs``."""
    df = pairs.join(profile, on="CANDIDATE_ID", how="left").join(
        jobs_req, on="Job_id", how="left"
    )
    return df.with_columns(
        _subset("req_licenses", "licenses").alias("Card_match" + suffix),
        _subset("req_languages", "languages").alias("Languages_match" + suffix),
        _subset("req_regulated", "regulated").alias(
            "Regulated_Trainings_match" + suffix
        ),
        _fraction("req_skills", "skills").alias("match_skills" + suffix),
        _distance(pl.col("distance_km")).alias("Distance_match" + suffix),
    ).select(pairs.columns + [c + suffix for c in RULE_COMPONENTS])
