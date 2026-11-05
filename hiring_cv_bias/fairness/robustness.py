"""Robustness of the selection to the noise of the profile: why the parser's errors flip
more decisions for some groups (section 3.6 of the notebooks).

The outcome of a pair is its position in the job's list, so an error of the profile flips
it only when the pair's score is closer to the cut than the error is large.
``margin_to_cutoff`` is that distance; ``margin_table`` and ``error_by_margin_band``
describe it per group and ``standardized_error`` takes it out of the comparison between
groups (``share_where_gap_significant`` keeps the share it explains only where the gap is
significant). ``term_spread`` says which term of the score separates the candidates of a
list, per group; ``list_class`` sorts the lists by whether the job's required skills
separate their candidates (``match_skills`` is the term that loses most of its spread in
the South), with ``list_classes_table`` and ``error_by_class``.
``perturbation_flips`` measures the stability of the decisions directly, as the share
that change under random noise on ``Experience``."""

from collections.abc import Sequence

import numpy as np
import polars as pl

from hiring_cv_bias.fairness.metrics import (
    ATTRIBUTES,
    GROUP_ORDER,
    per_group,
    share,
    with_weights,
)
from hiring_cv_bias.fairness.protocol import (
    CUTOFFS,
    SEED,
    decided_by_draw,
    rank_within_job,
)
from hiring_cv_bias.reverse_matching.scoring import (
    COMPONENTS,
    RULE_COMPONENTS,
    WEIGHTS,
    score_expr,
)

# a difference of spread below this (in points of score) is no difference: the relative
# change of ``term_spread`` is 0
MIN_SPREAD = 0.01
# the distance from the cut, in points of score, in bands
MARGIN_BANDS = ("0 (tie)", "(0, 0.1]", "(0.1, 0.3]", "(0.3, 0.6]", "> 0.6")
LIST_CLASSES = (
    "none required",
    "required, same for all",
    "required, discriminate",
)


def margin_to_cutoff(version: str = "O", outcome: str = "top_half") -> pl.Expr:
    """The pair's score under ``version`` minus the score of the last selected candidate
    of its list (the cutoff score): positive inside the selection, negative outside; its
    absolute value is how far the pair is from the cut, the confidence of a ranking
    with fixed places. Needs the ranks of ``add_outcomes``."""
    score, rank = pl.col(f"score_{version}"), pl.col(f"rank_{version}")
    cutoff_score = score.filter(rank == CUTOFFS[outcome]).first().over("Job_id")
    return score - cutoff_score


def margin_band(margin: pl.Expr) -> pl.Expr:
    """The band of ``MARGIN_BANDS`` of a margin (signed or absolute)."""
    m = margin.abs()
    return (
        pl.when(m == 0)
        .then(pl.lit(MARGIN_BANDS[0]))
        .when(m <= 0.1)
        .then(pl.lit(MARGIN_BANDS[1]))
        .when(m <= 0.3)
        .then(pl.lit(MARGIN_BANDS[2]))
        .when(m <= 0.6)
        .then(pl.lit(MARGIN_BANDS[3]))
        .otherwise(pl.lit(MARGIN_BANDS[4]))
    )


def _with_margin(df: pl.DataFrame, version: str, outcome: str) -> pl.DataFrame:
    """``df`` with the absolute margin and its band under ``version``; window
    expressions are evaluated here because a ``group_by`` cannot hold them."""
    margin = margin_to_cutoff(version, outcome).abs()
    return with_weights(df).with_columns(
        margin.alias("_margin"), margin_band(margin).alias("_band")
    )


def margin_table(
    df: pl.DataFrame,
    attributes: Sequence[str] = ATTRIBUTES,
    version: str = "O",
    draws: Sequence[str] = ("adecco", "O"),
    outcome: str = "top_half",
) -> pl.DataFrame:
    """How close to the cut the pairs of every group are under ``version``: the median
    absolute margin, the share of pairs within 0.1 points of the cut, and, for every score
    in ``draws``, the share of pairs whose outcome the tie-break decides
    (``draw_<score>``)."""
    d = _with_margin(df, version, outcome).with_columns(
        decided_by_draw(pl.col(f"score_{s}"), outcome).alias(f"_draw_{s}")
        for s in draws
    )
    return per_group(
        d,
        attributes,
        [
            pl.col("_margin").median().alias("margin_median"),
            share(pl.col("_margin") <= 0.1).alias("within_0.1"),
            *[share(pl.col(f"_draw_{s}")).alias(f"draw_{s}") for s in draws],
        ],
    )


def _wrong(decision: str, label: str, outcome: str) -> pl.Expr:
    return pl.col(f"{outcome}_{decision}") != pl.col(f"{outcome}_{label}")


def error_by_margin_band(
    df: pl.DataFrame,
    attribute: str = "Location",
    decision: str = "O",
    label: str = "C",
    outcome: str = "top_half",
) -> pl.DataFrame:
    """Per band of the margin under ``decision`` and value of ``attribute``: the pairs
    and the share whose outcome differs between ``decision`` and ``label``."""
    d = _with_margin(df, decision, outcome)
    order = pl.col("band").replace_strict(
        {b: i for i, b in enumerate(MARGIN_BANDS)}, return_dtype=pl.Int64
    )
    return (
        d.group_by(pl.col("_band").alias("band"), attribute)
        .agg(
            pl.len().alias("pairs"),
            share(_wrong(decision, label, outcome)).alias("wrong"),
        )
        .sort(order, attribute)
    )


def standardized_error(
    df: pl.DataFrame,
    attribute: str = "Location",
    reference: str = "NORTH",
    decision: str = "O",
    label: str = "C",
    outcome: str = "top_half",
) -> pl.DataFrame:
    """Per value of ``attribute``: the error rate, the share of pairs whose outcome
    differs between ``decision`` and ``label`` (``error``); the same rate if the group
    had the reference group's distribution of margins (``error_std``: the group's rate
    in every band, weighed by the share of the reference's pairs in that band); the
    gaps of both from the reference (``gap``, ``gap_std``); and what the standardization
    removes from the gap, the part due to the margins (``explained``, ``gap − gap_std``),
    also as a share of the gap (``explained_share``, null for the reference). Computed on
    the same pairs, the two gaps move together across bootstrap replicates, so
    ``explained`` has a narrower interval than either."""
    d = _with_margin(df, decision, outcome)
    wrong = _wrong(decision, label, outcome)
    rates = d.group_by(attribute, "_band").agg(share(wrong).alias("rate"))
    weights = (
        d.filter(pl.col(attribute) == reference)
        .group_by("_band")
        .agg(pl.col("w").sum().alias("ref_w"))
        .with_columns(pl.col("ref_w") / pl.col("ref_w").sum())
    )
    standardized = (
        rates.join(weights, on="_band", how="inner")
        .group_by(attribute)
        .agg((pl.col("rate") * pl.col("ref_w")).sum().alias("error_std"))
    )
    crude = d.group_by(attribute).agg(
        pl.len().alias("pairs"), share(wrong).alias("error")
    )
    out = crude.join(standardized, on=attribute, how="left").sort(attribute)
    ref = pl.col(attribute) == reference

    def gap(col: str) -> pl.Expr:
        return pl.col(col) - pl.col(col).filter(ref).first()

    return (
        out.with_columns(gap("error").alias("gap"), gap("error_std").alias("gap_std"))
        .with_columns((pl.col("gap") - pl.col("gap_std")).alias("explained"))
        .with_columns(
            pl.when(pl.col("gap") != 0)
            .then(pl.col("explained") / pl.col("gap"))
            .alias("explained_share")
        )
        .select(
            pl.lit(attribute).alias("attribute"),
            pl.col(attribute).alias("group"),
            "pairs",
            "error",
            "error_std",
            "gap",
            "gap_std",
            "explained",
            "explained_share",
        )
    )


def share_where_gap_significant(
    table: pl.DataFrame, share: str = "explained_share", gap: str = "gap"
) -> pl.DataFrame:
    """A bootstrapped ``standardized_error`` table with ``share`` and its bounds set to
    null where the interval of ``gap`` holds 0: the share of a gap that may not exist has
    no meaning, and its replicates divide by gaps near 0 of either sign."""
    holds_zero = (pl.col(f"{gap}_lo") <= 0) & (pl.col(f"{gap}_hi") >= 0)
    return table.with_columns(
        pl.when(~holds_zero).then(pl.col(c)).alias(c)
        for c in (share, f"{share}_lo", f"{share}_hi")
    )


def term_column(component: str, version: str) -> str:
    """The column of ``component`` under ``version``: ``Interest`` comes from the candidate's
    choices and has one copy, ``Experience`` is the estimate ``Experience_hat``."""
    if component == "Interest":
        return "Interest"
    if component == "Experience":
        return f"Experience_hat_{version}"
    return f"{component}_{version}"


def term_spread(
    df: pl.DataFrame,
    version: str = "O",
    attribute: str = "Location",
    reference: str = "NORTH",
    group: str = "SOUTH",
) -> pl.DataFrame:
    """How much each term of the score separates the candidates of a list, per value of
    ``attribute``: the standard deviation within the job's list of every weighted
    component (weight × component, in points of score) and of the score, averaged over
    the pairs of the group. One row per term, one column per group in display order, and
    the relative change of ``group`` from ``reference`` (``change``, −0.5 when the
    spread halves; 0 when the two differ by less than ``MIN_SPREAD``, so that a term
    that separates nobody in either group does not show a change)."""
    terms = {c: WEIGHTS[c] * pl.col(term_column(c, version)) for c in COMPONENTS}
    terms["score"] = pl.col(f"score_{version}")
    d = df.with_columns(
        e.std().over("Job_id").alias(f"_sd_{t}") for t, e in terms.items()
    )
    by_group = d.group_by(attribute).agg(pl.col(f"_sd_{t}").mean() for t in terms)
    groups = sorted(by_group[attribute].to_list(), key=lambda g: GROUP_ORDER.index(g))
    out = pl.DataFrame(
        {
            "term": list(terms),
            **{
                g: [
                    by_group.filter(pl.col(attribute) == g)[f"_sd_{t}"][0]
                    for t in terms
                ]
                for g in groups
            },
        }
    )
    return out.with_columns(
        pl.when((pl.col(group) - pl.col(reference)).abs() < MIN_SPREAD)
        .then(0.0)
        .otherwise(pl.col(group) / pl.col(reference) - 1)
        .alias("change")
    )


def list_class(skills_required: pl.Expr, version: str = "O") -> pl.Expr:
    """Which of the ``LIST_CLASSES`` a pair's list belongs to: the job requires no skill
    (``skills_required`` false), it requires skills that give every candidate of the
    list the same ``match_skills`` (nobody, or everybody alike, has them), or it requires
    skills that separate the candidates of the list."""
    same = pl.col(f"match_skills_{version}").n_unique().over("Job_id") == 1
    return (
        pl.when(~skills_required)
        .then(pl.lit(LIST_CLASSES[0]))
        .when(same)
        .then(pl.lit(LIST_CLASSES[1]))
        .otherwise(pl.lit(LIST_CLASSES[2]))
    )


def list_classes_table(
    df: pl.DataFrame,
    classes: pl.Expr,
    version: str = "O",
    attribute: str = "Location",
    outcome: str = "top_half",
) -> pl.DataFrame:
    """Per value of ``attribute`` (the value of most of the list's candidates, the first
    in sorted order when two tie) and class of ``classes``: the lists and their share of the region's lists, the pairs, and the
    median distance of a pair from the cut under ``version``."""
    d = _with_margin(df, version, outcome).with_columns(classes.alias("class"))
    # the most frequent value of the list; a tie goes to the first in sorted order
    lists = d.group_by("Job_id").agg(
        pl.col(attribute).mode().sort().first().alias(attribute),
        pl.col("class").first(),
    )
    by_list = (
        lists.group_by(attribute, "class")
        .agg(pl.len().alias("lists"))
        .with_columns(
            (pl.col("lists") / pl.col("lists").sum().over(attribute)).alias(
                "share_of_lists"
            )
        )
    )
    by_pair = d.group_by(attribute, "class").agg(
        pl.len().alias("pairs"), pl.col("_margin").median().alias("margin_median")
    )
    return by_list.join(by_pair, on=[attribute, "class"]).sort(attribute, "class")


def error_by_class(
    df: pl.DataFrame,
    classes: pl.Expr,
    attribute: str = "Location",
    reference: str = "NORTH",
    group: str = "SOUTH",
    decision: str = "O",
    label: str = "C",
    outcome: str = "top_half",
) -> pl.DataFrame:
    """For every class of ``classes``: the pairs of ``group`` in the class, the share of
    pairs whose outcome differs between ``decision`` and ``label`` for ``reference`` and
    for ``group`` (``error_<reference>``, ``error_<group>``, lower case), and the gap
    between the two (``gap``); classes in the order of ``LIST_CLASSES``."""
    d = with_weights(df).with_columns(classes.alias("class"))
    wrong = _wrong(decision, label, outcome)
    is_ref, is_group = pl.col(attribute) == reference, pl.col(attribute) == group
    err_ref, err_group = f"error_{reference.lower()}", f"error_{group.lower()}"
    order = pl.col("class").replace_strict(
        {c: i for i, c in enumerate(LIST_CLASSES)}, return_dtype=pl.Int64
    )
    return (
        d.group_by("class")
        .agg(
            is_group.sum().alias("pairs"),
            share(wrong, is_ref).alias(err_ref),
            share(wrong, is_group).alias(err_group),
        )
        .filter(pl.col("pairs") > 0)
        .with_columns((pl.col(err_group) - pl.col(err_ref)).alias("gap"))
        .sort(order)
    )


def perturbation_flips(
    df: pl.DataFrame,
    version: str = "O",
    sigmas: Sequence[float] = (0.05, 0.1),
    n_draws: int = 20,
    by: Sequence[str] = ("Location",),
    seed: int = SEED,
    outcome: str = "top_half",
) -> pl.DataFrame:
    """How stable the decisions of ``version`` are: for every ``sigma``, Gaussian noise
    of that standard deviation is added to the estimated ``Experience`` of every pair
    (kept within 0 and 1), the score is recomputed with the formula, the lists are
    re-ranked with the same tie-break, and the share of pairs whose outcome changes is
    averaged over ``n_draws`` draws, per value of the columns ``by``. The same noise
    everywhere: a group whose decisions flip more often has them closer to the cut."""
    rng = np.random.default_rng(seed)
    base = pl.col(f"{outcome}_{version}")
    columns: dict[str, str | pl.Expr] = {c: f"{c}_{version}" for c in RULE_COMPONENTS}
    columns["Experience"] = (
        pl.col(f"Experience_hat_{version}") + pl.col("_noise")
    ).clip(0.0, 1.0)
    rank = rank_within_job(score_expr(WEIGHTS, columns)).alias("_rank")
    flipped = ((pl.col("_rank") <= CUTOFFS[outcome]) != base).cast(pl.Float64)
    rows = []
    for sigma in sigmas:
        flips = np.zeros(df.height)
        for _ in range(n_draws):
            # the noise and the rank are columns, so that the window sees them per job
            noisy = df.with_columns(
                pl.Series("_noise", rng.normal(0.0, sigma, df.height))
            ).with_columns(rank)
            flips += noisy.select(flipped).to_series().to_numpy()
        rows.append(
            df.with_columns(pl.Series("_flip", flips / n_draws))
            .group_by(list(by))
            .agg(pl.len().alias("pairs"), pl.col("_flip").mean().alias("flipped"))
            .with_columns(pl.lit(float(sigma)).alias("sigma"))
            .select("sigma", *by, "pairs", "flipped")
            .sort(list(by))
        )
    return pl.concat(rows)
