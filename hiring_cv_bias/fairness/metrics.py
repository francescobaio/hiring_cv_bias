"""Group fairness of the selection under the parser's profile (O) and the corrected one (C):
the questions of phase 3 of the notebooks.

What the correction changes and for whom (``profile_corrections``, ``parser_changes``,
``outcome_change_sources``); whether every group is selected at the same rate
(``selection_rates``, independence); whether the parser's errors have the same rates in
every group (``parser_effect``, separation); confidence intervals for all of it
(``bootstrap``) and tables readable in percent (``for_display``).

Every statistic is computed on the protocol pairs with their outcomes (``add_outcomes``)
and reported per group of each sensitive attribute, in long form, each group against the
reference group of its attribute. The statistics are weighted means: the weight column
``w`` is 1 for the point estimate (added when absent) and the number of times the pair's
candidate was drawn in a bootstrap replicate, so the same functions serve both."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import polars as pl

from hiring_cv_bias.fairness.protocol import (
    COMPARED_VERSIONS,
    SCORES,
    SEED,
    add_outcomes,
)
from hiring_cv_bias.reverse_matching.audit import differs
from hiring_cv_bias.reverse_matching.profiles import PARSER_FIELDS

ATTRIBUTES = ("Gender", "Location", "group")
# the group every other group of the attribute is compared with
REFERENCE = {"Gender": "Male", "Location": "NORTH", "group": "Male_NORTH"}
# display order of the groups; a group outside the list goes last
GROUP_ORDER = [
    "Female",
    "Male",
    "NORTH",
    "CENTER",
    "SOUTH",
    "Female_NORTH",
    "Male_NORTH",
    "Female_CENTER",
    "Male_CENTER",
    "Female_SOUTH",
    "Male_SOUTH",
]
_ORDER = {g: i for i, g in enumerate(GROUP_ORDER)}
N_BOOT = 1000

# whether correcting the parser (O → C) changes the pair's license match, language match
# and estimated Experience (by more than 0.01), with the tolerances of the data audit
COMPONENT_CHANGED = {
    "Card_match": differs("Card_match", "O", "C"),
    "Languages_match": differs("Languages_match", "O", "C"),
    "Experience": differs("Experience_hat", "O", "C"),
}
SCORE_CHANGED = differs("score", "O", "C")

Stat = Callable[[pl.DataFrame], pl.DataFrame]


def _share(event: pl.Expr, among: pl.Expr | None = None) -> pl.Expr:
    """Share of the pairs where ``event`` holds, among those where ``among`` holds (all the
    pairs by default), as a weighted mean with the weights ``w``. Every rate of the module
    is built from it."""
    w = pl.col("w") if among is None else pl.col("w") * among.cast(pl.Float64)
    return (w * event.cast(pl.Float64)).sum() / w.sum()


def _per_group(
    df: pl.DataFrame, attributes: Sequence[str], aggs: Sequence[pl.Expr]
) -> pl.DataFrame:
    """Compute ``aggs`` for every group of every attribute and stack the results in one long
    table: one row per (attribute, group), in display order, with the number of pairs.
    Adds unit weights when ``df`` has no ``w`` column, so the aggregations can rely on it."""
    if "w" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("w"))
    order = pl.col("group").replace_strict(
        _ORDER, default=len(_ORDER), return_dtype=pl.Int64
    )
    return (
        pl.concat(
            df.group_by(a)
            .agg(pl.len().alias("pairs"), *aggs)
            .rename({a: "group"})
            .with_columns(pl.lit(a).alias("attribute"))
            .select("attribute", "group", pl.exclude("attribute", "group"))
            for a in attributes
        )
        .with_columns(order.alias("_order"))
        .sort("_order", maintain_order=True)
        .drop("_order")
    )


def _reference(col: str) -> pl.Expr:
    """On every row of a ``_per_group`` table, the value of ``col`` for the reference group
    of the row's attribute, so that a gap can be written as ``col - _reference(col)``."""
    is_ref = pl.any_horizontal(
        (pl.col("attribute") == a) & (pl.col("group") == g)
        for a, g in REFERENCE.items()
    )
    return pl.col(col).filter(is_ref).first().over("attribute")


def profile_corrections(
    profile_o: pl.DataFrame, profile_c: pl.DataFrame, candidates: pl.Series
) -> pl.DataFrame:
    """How much the correction adds to the profiles of ``candidates``: per field read by the
    parser, the number and share of candidates who gain at least one item, and the items
    added in all."""
    # ``implode`` turns the Series into one list, the form ``is_in`` requires from polars 1.30
    df = profile_o.join(profile_c, on="CANDIDATE_ID", suffix="_C").filter(
        pl.col("CANDIDATE_ID").is_in(candidates.implode())
    )
    added = {
        f: pl.col(f"{f}_C").list.len() - pl.col(f).list.len() for f in PARSER_FIELDS
    }
    return pl.concat(
        df.select(
            pl.lit(f).alias("field"),
            (n > 0).sum().alias("candidates_corrected"),
            (n > 0).mean().alias("share_of_candidates"),
            n.sum().alias("items_added"),
        )
        for f, n in added.items()
    )


def parser_changes(
    df: pl.DataFrame, attributes: Sequence[str] = ATTRIBUTES
) -> pl.DataFrame:
    """How often the correction touches a pair: per group, the share of pairs whose license
    match, language match, estimated ``Experience`` (beyond the tolerance of the data
    audit) or score differs between O and C."""
    changed = {**COMPONENT_CHANGED, "score": SCORE_CHANGED}
    return _per_group(df, attributes, [_share(e).alias(k) for k, e in changed.items()])


def outcome_change_sources(
    df: pl.DataFrame, attributes: Sequence[str] = ATTRIBUTES, outcome: str = "top_half"
) -> pl.DataFrame:
    """Why pairs change outcome. Among the pairs whose outcome differs between O and C, per
    group: the share whose own license match, language match or ``Experience`` changed (a
    pair can count in more than one); ``other``, the share whose score moved only through
    an ``Experience`` shift within the tolerance; and ``score_unchanged``, the share whose
    own score did not move at all, so that they changed outcome only because a competitor
    of the same job was corrected."""
    changed = df.filter(pl.col(f"{outcome}_O") != pl.col(f"{outcome}_C"))
    any_component = pl.any_horizontal(*COMPONENT_CHANGED.values())
    return _per_group(
        changed,
        attributes,
        [_share(e).alias(k) for k, e in COMPONENT_CHANGED.items()]
        + [
            _share(SCORE_CHANGED & ~any_component).alias("other"),
            _share(~SCORE_CHANGED).alias("score_unchanged"),
        ],
    ).rename({"pairs": "changed_pairs"})


def selection_rates(
    df: pl.DataFrame,
    attributes: Sequence[str] = ATTRIBUTES,
    versions: Sequence[str] = COMPARED_VERSIONS,
    outcome: str = "top_half",
) -> pl.DataFrame:
    """Independence. The selection rate of every group under each score in ``versions``
    (``rate_<version>``), with its gap from the reference group as a difference
    (``spd_<version>``, statistical parity difference, 0 at parity) and as a ratio
    (``di_<version>``, disparate impact, 1 at parity)."""
    rates = _per_group(
        df,
        attributes,
        [_share(pl.col(f"{outcome}_{v}")).alias(f"rate_{v}") for v in versions],
    )
    return rates.with_columns(
        *[
            (pl.col(f"rate_{v}") - _reference(f"rate_{v}")).alias(f"spd_{v}")
            for v in versions
        ],
        *[
            (pl.col(f"rate_{v}") / _reference(f"rate_{v}")).alias(f"di_{v}")
            for v in versions
        ],
    )


def parser_effect(
    df: pl.DataFrame, attributes: Sequence[str] = ATTRIBUTES, outcome: str = "top_half"
) -> pl.DataFrame:
    """Separation and sufficiency. The parser's errors per group, taking the outcome with
    the corrected profile as the label and the outcome with the parser's profile as the
    decision: the selection rates with O and C and their change (``rate_change``); the
    false negative rate of O (``fnr``, pairs selected with C that O leaves out) and its
    false positive rate (``fpr``, pairs not selected with C that O selects), whose
    equality across groups is separation; the positive and negative predictive values
    (``ppv``, pairs O selects that C confirms; ``npv``, pairs O rejects that C confirms),
    whose equality is sufficiency; and the gap of each rate from the reference group
    (``<rate>_diff``, 0 when the criterion holds)."""
    d, y = pl.col(f"{outcome}_O"), pl.col(f"{outcome}_C")
    effect = _per_group(
        df,
        attributes,
        [
            _share(d).alias("rate_O"),
            _share(y).alias("rate_C"),
            (_share(y) - _share(d)).alias("rate_change"),
            _share(~d, among=y).alias("fnr"),
            _share(d, among=~y).alias("fpr"),
            _share(y, among=d).alias("ppv"),
            _share(~y, among=~d).alias("npv"),
        ],
    )
    return effect.with_columns(
        (pl.col(c) - _reference(c)).alias(f"{c}_diff")
        for c in ("fnr", "fpr", "ppv", "npv")
    )


def bootstrap(
    df: pl.DataFrame,
    stats: Mapping[str, Stat],
    n_boot: int = N_BOOT,
    seed: int = SEED,
    level: float = 0.95,
    score_cols: Sequence[str] = SCORES,
) -> dict[str, pl.DataFrame]:
    """Confidence intervals for the tables of ``stats``, by resampling candidates.

    The point estimate of each statistic is its value on ``df`` as given. Each of the
    ``n_boot`` replicates draws the candidates with replacement (a candidate drawn k times
    weighs k, and all their pairs move together), redraws the tie-break of ``score_cols``
    and recomputes the statistics; the job lists and the decisions inside them stay as
    observed. The interval of every float column is the central ``level`` range of the
    replicates, returned as ``<col>_lo`` and ``<col>_hi`` next to the point estimate;
    counts and keys are those of the point estimate."""
    df = df.sort("Job_id", "CANDIDATE_ID")
    candidate = df["CANDIDATE_ID"].rank("dense").cast(pl.Int64).to_numpy() - 1
    n_candidates = int(candidate.max()) + 1
    rng = np.random.default_rng(seed)

    point = {k: f(df) for k, f in stats.items()}
    reps: dict[str, list[pl.DataFrame]] = {k: [] for k in stats}
    for _ in range(n_boot):
        d = add_outcomes(df, score_cols, seed=int(rng.integers(2**32)))
        drawn = np.bincount(
            rng.integers(0, n_candidates, n_candidates), minlength=n_candidates
        )
        d = d.with_columns(pl.Series("w", drawn[candidate], dtype=pl.Float64))
        for k, f in stats.items():
            reps[k].append(f(d))

    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    out = {}
    for k, p in point.items():
        keys = [c for c, t in p.schema.items() if t == pl.String]
        values = [c for c, t in p.schema.items() if t == pl.Float64]
        ci = (
            pl.concat(reps[k])
            .group_by(keys)
            .agg(
                *[pl.col(c).quantile(lo).alias(f"{c}_lo") for c in values],
                *[pl.col(c).quantile(hi).alias(f"{c}_hi") for c in values],
            )
        )
        others = [c for c in p.columns if c not in values]
        out[k] = p.join(ci, on=keys, how="left", maintain_order="left").select(
            *others, *[x for c in values for x in (c, f"{c}_lo", f"{c}_hi")]
        )
    return out


def for_display(
    table: pl.DataFrame,
    percent: Sequence[str] | None = None,
    ratios: Sequence[str] = (),
    digits: int = 1,
) -> pl.DataFrame:
    """Make a table readable: the shares in ``percent`` (by default every float column that is
    not a ratio) become percentages with ``digits`` decimals, the ``ratios`` keep one more
    decimal, the other float columns are dropped, and a column with bootstrap bounds
    becomes the string ``value [lo, hi]`` with fixed decimals."""
    if percent is None:
        percent = [
            c
            for c, t in table.schema.items()
            if t == pl.Float64 and not c.endswith(("_lo", "_hi")) and c not in ratios
        ]
    scale = {c: (100, digits) for c in percent} | {c: (1, digits + 1) for c in ratios}

    def number(col: str, source: str) -> pl.Expr:
        factor, d = scale[col]
        x = (factor * pl.col(source)).round(d)
        return pl.when(x == 0).then(0.0).otherwise(x)  # no "-0.0"

    def text(col: str, source: str) -> pl.Expr:
        return number(col, source).cast(pl.Decimal(scale=scale[col][1])).cast(pl.String)

    def show(c: str) -> pl.Expr:
        if f"{c}_lo" in table.columns:
            bounds = [text(c, s) for s in (c, f"{c}_lo", f"{c}_hi")]
            return pl.format("{} [{}, {}]", *bounds)
        return number(c, c)

    return table.select(
        show(c).alias(c) if c in scale else pl.col(c)
        for c, t in table.schema.items()
        if c in scale or t != pl.Float64
    )
