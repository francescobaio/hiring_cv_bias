"""What counts as a selection: the evaluation protocol, fixed before any fairness metric is
computed (section 2.1 of the notebooks).

A score means something only against the other candidates of the same job, so the
outcome of a (candidate, job) pair is its position within the job's list: selected when in
the top half (``top_half``, the primary outcome) or among the first three (``top3``, the
stricter one). Ties are broken by a random draw shared by every score, so that two
versions of the pipeline can differ only through the scores. The analysis keeps the
Female/Male candidates and the jobs with at least ``MIN_CANDIDATES`` of them, and ranks
each list four times: with Adecco's own score and with the recovered formula on the three
profile versions P, O and C."""

from collections.abc import Sequence

import numpy as np
import polars as pl

from hiring_cv_bias.config import PAIRS_PATH

MIN_CANDIDATES = 5
TOP_K = 3
SEED = 0
GENDERS = ("Female", "Male")
VERSIONS = ("adecco", "P", "O", "C")
SCORES = [f"score_{v}" for v in VERSIONS]
# the scores whose selections phase 3 compares: the real system and the two parser versions
COMPARED_VERSIONS = ("adecco", "O", "C")
# the outcomes: name -> last selected position within the job's list of ``n_in_job`` rows
CUTOFFS = {"top_half": (pl.col("n_in_job") / 2).ceil(), f"top{TOP_K}": pl.lit(TOP_K)}


def add_group(df: pl.DataFrame) -> pl.DataFrame:
    """Add the intersectional attribute ``group``, gender and macro-region together
    (e.g. ``Female_SOUTH``)."""
    return df.with_columns(
        pl.concat_str([pl.col("Gender"), pl.col("Location")], separator="_").alias(
            "group"
        )
    )


def load_pairs(path: str = PAIRS_PATH) -> pl.DataFrame:
    """Load the pairs built by phase 1, ready for the protocol: with ``group`` and with
    Adecco's ``match_score`` exposed as ``score_adecco`` next to ``score_P/O/C``."""
    return add_group(pl.read_parquet(path)).with_columns(
        pl.col("match_score").alias("score_adecco")
    )


def filter_protocol(
    df: pl.DataFrame,
    min_candidates: int = MIN_CANDIDATES,
    genders: Sequence[str] = GENDERS,
) -> pl.DataFrame:
    """Restrict the pairs to the ones the protocol evaluates: candidates of the ``genders``
    with enough of them for group statistics, in job lists long enough for a position to
    mean something (at least ``min_candidates`` after the gender filter)."""
    out = df.filter(pl.col("Gender").is_in(list(genders)))
    return out.filter(pl.len().over("Job_id") >= min_candidates)


def rank_within_job(score: pl.Expr) -> pl.Expr:
    """Position of the pair among the candidates of its job under ``score`` (1 is best),
    ties broken by the pair's ``tie_break``."""
    return (
        pl.struct(score, pl.col("tie_break"))
        .rank(method="ordinal", descending=True)
        .over("Job_id")
        .cast(pl.Int32)
    )


def decided_by_draw(score: pl.Expr, outcome: str = "top_half") -> pl.Expr:
    """Whether the pair's outcome depends on the tie-break: tied under ``score`` with another
    candidate of the job across the cutoff."""
    cutoff = CUTOFFS[outcome]
    best = score.rank(method="min", descending=True).over("Job_id")
    worst = score.rank(method="max", descending=True).over("Job_id")
    return (best <= cutoff) & (worst > cutoff)


def add_outcomes(
    df: pl.DataFrame, score_cols: Sequence[str] = SCORES, seed: int = SEED
) -> pl.DataFrame:
    """Turn scores into decisions. For every ``score_X`` column, rank the candidates of each
    job (``rank_X``, 1 is best) and derive one boolean per outcome of ``CUTOFFS``
    (``top_half_X``, ``top3_X``). Ties are broken by ``tie_break``, one random number per
    pair drawn from ``seed`` on the pairs sorted by job and candidate, so the draw depends
    neither on the row order nor on the score: a pair gets a different outcome under two
    scores only because the scores differ. Every job selects exactly ⌈n/2⌉ candidates for
    ``top_half``; ``n_in_job`` is the length of the list."""
    out = df.sort("Job_id", "CANDIDATE_ID").with_columns(
        pl.len().over("Job_id").alias("n_in_job"),
        pl.Series("tie_break", np.random.default_rng(seed).random(df.height)),
    )
    suffixes = [c.removeprefix("score") for c in score_cols]
    out = out.with_columns(
        rank_within_job(pl.col(c)).alias("rank" + s)
        for c, s in zip(score_cols, suffixes, strict=True)
    )
    return out.with_columns(
        (pl.col("rank" + s) <= cutoff).alias(name + s)
        for s in suffixes
        for name, cutoff in CUTOFFS.items()
    )


def protocol_sizes(df: pl.DataFrame, by: str = "group") -> pl.DataFrame:
    """Size of the evaluation set: pairs, jobs and candidates for every value of ``by``, plus
    a total row."""
    sizes = [
        pl.len().alias("pairs"),
        pl.col("Job_id").n_unique().alias("jobs"),
        pl.col("CANDIDATE_ID").n_unique().alias("candidates"),
    ]
    per_group = df.group_by(by).agg(sizes).sort(by)
    total = df.select(pl.lit("all").alias(by), *sizes)
    return pl.concat([per_group, total])


def list_composition(
    df: pl.DataFrame, attributes: Sequence[str] = ("Gender", "Location")
) -> pl.DataFrame:
    """How mixed the lists are. Per attribute: the lists whose candidates all have the same
    value (``one_value``), their share of the lists and the share of the pairs that are in
    them. A list with one value gives that value all its ⌈n/2⌉ selections whatever the
    score, so its selection rate there does not depend on the score."""
    lists = df.group_by("Job_id").agg(
        pl.len().alias("n"), *[(pl.col(a).n_unique() == 1).alias(a) for a in attributes]
    )
    return pl.concat(
        [
            lists.select(
                pl.lit(a).alias("attribute"),
                pl.len().alias("lists"),
                pl.col(a).sum().alias("one_value"),
                pl.col(a).mean().alias("share_of_lists"),
                (pl.col("n").filter(pl.col(a)).sum() / pl.col("n").sum()).alias(
                    "share_of_pairs"
                ),
            )
            for a in attributes
        ]
    )


def outcome_summary(
    df: pl.DataFrame, versions: Sequence[str] = VERSIONS, outcome: str = "top_half"
) -> pl.DataFrame:
    """How much the outcome depends on the draw and on the score. Per version: the share of
    pairs whose outcome is decided by the tie-break (tied with another candidate of the
    job across the cutoff) and the share that gets the same outcome as under the first
    version in ``versions``, the agreement with it."""
    rows = []
    for v in versions:
        rows.append(
            df.select(
                pl.lit(v).alias("version"),
                decided_by_draw(pl.col(f"score_{v}"), outcome)
                .mean()
                .alias("decided_by_draw"),
                (pl.col(f"{outcome}_{v}") == pl.col(f"{outcome}_{versions[0]}"))
                .mean()
                .alias(f"agree_{versions[0]}"),
            )
        )
    return pl.concat(rows)
