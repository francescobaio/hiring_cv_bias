"""The evaluation protocol, fixed before any fairness metric is computed (section 2.1 of
the notebook).

* **Unit**: the (candidate, job) pair of ``data/ethics/pairs.parquet``.
* **Outcome**: computed *within* the job's list, because a score is only comparable among
  the candidates of the same job. ``top_half`` (rank ≤ ⌈n/2⌉) is the primary outcome and
  ``top3`` (rank ≤ 3) the secondary one. Ties are broken by a random draw: every pair gets
  one random number, used by every score, so a pair changes outcome between two versions
  only because its score or its competitors' scores change.
* **Lists**: jobs with at least ``MIN_CANDIDATES`` Female/Male candidates.
* **Sensitive attributes**: ``Gender`` (Female/Male), ``Location`` (NORTH, CENTER, SOUTH)
  and their intersection ``group``.
* **Scores**: ``score_adecco`` (Adecco's ``match_score``) and ``score_P/O/C`` (the formula on
  each profile version).
"""

from collections.abc import Sequence

import numpy as np
import polars as pl

from hiring_cv_bias.reverse_matching.build import PAIRS_PATH

MIN_CANDIDATES = 5
TOP_K = 3
SEED = 0
GENDERS = ("Female", "Male")
VERSIONS = ("adecco", "P", "O", "C")
SCORES = [f"score_{v}" for v in VERSIONS]


def add_group(df: pl.DataFrame) -> pl.DataFrame:
    """Intersectional attribute, e.g. ``Female_SOUTH``."""
    return df.with_columns(
        pl.concat_str([pl.col("Gender"), pl.col("Location")], separator="_").alias(
            "group"
        )
    )


def load_pairs(path: str = PAIRS_PATH) -> pl.DataFrame:
    """The pairs of the build step, with ``group`` and Adecco's score as ``score_adecco``."""
    return add_group(pl.read_parquet(path)).with_columns(
        pl.col("match_score").alias("score_adecco")
    )


def filter_protocol(
    df: pl.DataFrame,
    min_candidates: int = MIN_CANDIDATES,
    genders: Sequence[str] = GENDERS,
) -> pl.DataFrame:
    """Keep the Female/Male candidates and the jobs with enough of them *after* that filter."""
    out = df.filter(pl.col("Gender").is_in(list(genders)))
    out = out.with_columns(pl.len().over("Job_id").alias("n_in_job"))
    return out.filter(pl.col("n_in_job") >= min_candidates)


def add_outcomes(
    df: pl.DataFrame,
    score_cols: Sequence[str] = SCORES,
    top_k: int = TOP_K,
    seed: int = SEED,
) -> pl.DataFrame:
    """Within-job position and outcomes of every score column: for ``score_X`` the
    columns ``rank_X``, ``top_half_X`` and ``top<k>_X``. Ties are broken by ``tie_break``,
    one random number per pair shared by every score; it is drawn on the pairs sorted by
    job and candidate, so it does not depend on the order of ``df``. Every job selects
    exactly ⌈n/2⌉ candidates for ``top_half``. ``n_in_job`` is recomputed on ``df``."""
    out = df.sort("Job_id", "CANDIDATE_ID").with_columns(
        pl.len().over("Job_id").alias("n_in_job"),
        pl.Series("tie_break", np.random.default_rng(seed).random(df.height)),
    )
    suffixes = [c.removeprefix("score") for c in score_cols]
    out = out.with_columns(
        pl.struct(c, "tie_break")
        .rank(method="ordinal", descending=True)
        .over("Job_id")
        .cast(pl.Int32)
        .alias("rank" + s)
        for c, s in zip(score_cols, suffixes, strict=True)
    )
    return out.with_columns(
        [
            (pl.col("rank" + s) <= (pl.col("n_in_job") / 2).ceil()).alias(
                "top_half" + s
            )
            for s in suffixes
        ]
        + [(pl.col("rank" + s) <= top_k).alias(f"top{top_k}" + s) for s in suffixes]
    )


def protocol_sizes(df: pl.DataFrame, by: str = "group") -> pl.DataFrame:
    """Pairs, jobs and candidates of every value of ``by``, plus the total."""
    per_group = df.group_by(by).agg(
        pl.len().alias("pairs"),
        pl.col("Job_id").n_unique().alias("jobs"),
        pl.col("CANDIDATE_ID").n_unique().alias("candidates"),
    )
    total = df.select(
        pl.lit("all").alias(by),
        pl.len().alias("pairs"),
        pl.col("Job_id").n_unique().alias("jobs"),
        pl.col("CANDIDATE_ID").n_unique().alias("candidates"),
    )
    return pl.concat([per_group.sort(by), total])


def _cutoff(outcome: str) -> pl.Expr:
    """Last selected position of ``top_half`` or ``top<k>``."""
    if outcome == "top_half":
        return (pl.col("n_in_job") / 2).ceil()
    return pl.lit(int(outcome.removeprefix("top")))


def outcome_summary(
    df: pl.DataFrame, versions: Sequence[str] = VERSIONS, outcome: str = "top_half"
) -> pl.DataFrame:
    """Per version: share of pairs whose outcome is decided by the draw (tied with another
    candidate of the job across the cutoff) and agreement with the outcome of the first
    version in ``versions``."""
    rows = []
    for v in versions:
        score = pl.col(f"score_{v}")
        best = score.rank(method="min", descending=True).over("Job_id")
        worst = score.rank(method="max", descending=True).over("Job_id")
        col = f"{outcome}_{v}"
        rows.append(
            df.select(
                pl.lit(v).alias("version"),
                ((best <= _cutoff(outcome)) & (worst > _cutoff(outcome)))
                .mean()
                .alias("decided_by_draw"),
                (pl.col(col) == pl.col(f"{outcome}_{versions[0]}"))
                .mean()
                .alias(f"agree_{versions[0]}"),
            )
        )
    return pl.concat(rows)
