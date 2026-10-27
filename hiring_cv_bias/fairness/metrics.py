"""Group fairness of the selection defined by the protocol (Phase 3 of the notebooks).

Every statistic is computed on the protocol pairs with their outcomes (``add_outcomes``)
and a weight column ``w``: 1 for the point estimate (the default when ``w`` is absent),
the number of times the pair's candidate was drawn in a bootstrap replicate. Groups are
the values of the sensitive attributes; the tables are in long form, one row per
(attribute, group)."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import polars as pl

from hiring_cv_bias.fairness.protocol import SCORES, SEED, add_outcomes
from hiring_cv_bias.reverse_matching.audit import differs
from hiring_cv_bias.reverse_matching.profiles import PARSER_FIELDS

ATTRIBUTES = ("Gender", "Location", "group")
# the group every other group is compared with
REFERENCE = {"Gender": "Male", "Location": "NORTH", "group": "Male_NORTH"}
# display order of the groups
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
    """Weighted share of the pairs where ``event`` holds, among those where ``among`` holds."""
    w = pl.col("w") if among is None else pl.col("w") * among.cast(pl.Float64)
    return (w * event.cast(pl.Float64)).sum() / w.sum()


def _per_group(
    df: pl.DataFrame, attributes: Sequence[str], aggs: Sequence[pl.Expr]
) -> pl.DataFrame:
    """``aggs`` for every group of every attribute, in long form and display order."""
    if "w" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("w"))
    return (
        pl.concat(
            df.group_by(a)
            .agg(pl.len().alias("pairs"), *aggs)
            .rename({a: "group"})
            .with_columns(pl.lit(a).alias("attribute"))
            .select("attribute", "group", pl.exclude("attribute", "group"))
            for a in attributes
        )
        .with_columns(pl.col("group").cast(pl.Enum(GROUP_ORDER)))
        .sort("group")
        .with_columns(pl.col("group").cast(pl.String))
    )


def _reference(col: str) -> pl.Expr:
    """On every row of a ``_per_group`` table, the value of ``col`` for the attribute's
    reference group (group names are unique across attributes)."""
    is_ref = pl.col("group").is_in(list(REFERENCE.values()))
    return pl.col(col).filter(is_ref).first().over("attribute")


def profile_corrections(
    profile_o: pl.DataFrame, profile_c: pl.DataFrame, candidates: pl.Series
) -> pl.DataFrame:
    """Per field read by the parser: how many of ``candidates`` the correction gives at
    least one item, their share, and how many items it adds in all."""
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
    """Per group: share of pairs where correcting the parser (O → C) changes the license
    and language match, the estimated ``Experience`` (by more than 0.01) and the score."""
    changed = {**COMPONENT_CHANGED, "score": SCORE_CHANGED}
    return _per_group(df, attributes, [_share(e).alias(k) for k, e in changed.items()])


def outcome_change_sources(
    df: pl.DataFrame, attributes: Sequence[str] = ATTRIBUTES, outcome: str = "top_half"
) -> pl.DataFrame:
    """Per group, among the pairs whose outcome changes from O to C: the share where the
    pair's own license match, language match or ``Experience`` (by more than 0.01)
    changes, and ``score_unchanged``, the share whose own score does not change at all
    (they move only because other candidates of the job do)."""
    changed = df.filter(pl.col(f"{outcome}_O") != pl.col(f"{outcome}_C"))
    return _per_group(
        changed,
        attributes,
        [_share(e).alias(k) for k, e in COMPONENT_CHANGED.items()]
        + [_share(~SCORE_CHANGED).alias("score_unchanged")],
    ).rename({"pairs": "changed_pairs"})


def selection_rates(
    df: pl.DataFrame,
    attributes: Sequence[str] = ATTRIBUTES,
    versions: Sequence[str] = ("adecco", "O", "C"),
    outcome: str = "top_half",
) -> pl.DataFrame:
    """Per group and score (independence):

    * ``rate_<version>``: the selection rate P(D = 1 | A = a);
    * ``spd_<version>``: statistical parity difference, the rate minus the rate of the
      attribute's reference group;
    * ``di_<version>``: disparate impact, the rate divided by the rate of the reference
      group."""
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
    """Per group, the outcome with the parser's profile (the decision D, from O) against
    the corrected one (the label Y, from C):

    * ``rate_O``, ``rate_C``: selection rates; ``rate_change`` = ``rate_C`` − ``rate_O``;
    * ``fnr``: false negative rate P(D = 0 | Y = 1, A = a), the share of the pairs
      selected with C that O leaves out;
    * ``fpr``: false positive rate P(D = 1 | Y = 0, A = a), the share of the pairs not
      selected with C that O selects;
    * ``fnr_diff``, ``fpr_diff``: the same minus the reference group (equalized odds asks
      both to be 0)."""
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
        ],
    )
    return effect.with_columns(
        (pl.col(c) - _reference(c)).alias(f"{c}_diff") for c in ("fnr", "fpr")
    )


def bootstrap(
    df: pl.DataFrame,
    stats: Mapping[str, Stat],
    n_boot: int = N_BOOT,
    seed: int = SEED,
    level: float = 0.95,
) -> dict[str, pl.DataFrame]:
    """Point estimates and percentile confidence intervals of every statistic in
    ``stats``. Each replicate draws a new tie-break, recomputes the outcomes on the real
    job lists, and resamples the candidates with replacement (a candidate drawn k times
    weighs k). Float columns get ``<col>_lo`` and ``<col>_hi``; string columns are the
    keys, and the other columns (counts) are those of the point estimate."""
    df = df.sort("Job_id", "CANDIDATE_ID")
    candidate = df["CANDIDATE_ID"].rank("dense").cast(pl.Int64).to_numpy() - 1
    n_candidates = int(candidate.max()) + 1
    rng = np.random.default_rng(seed)

    point = {k: f(df) for k, f in stats.items()}
    reps: dict[str, list[pl.DataFrame]] = {k: [] for k in stats}
    for _ in range(n_boot):
        d = add_outcomes(df, SCORES, seed=int(rng.integers(2**32)))
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
    """``table`` without its float columns except ``percent`` (default: every float column
    not in ``ratios``), shown in percent and rounded to ``digits``, and ``ratios``, rounded
    to ``digits + 1``. A column with bootstrap bounds becomes the string ``value [lo, hi]``."""
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

    def show(c: str) -> pl.Expr:
        if f"{c}_lo" in table.columns:
            bounds = [number(c, s) for s in (c, f"{c}_lo", f"{c}_hi")]
            return pl.format("{} [{}, {}]", *bounds)
        return number(c, c)

    return table.select(
        show(c).alias(c) if c in scale else pl.col(c)
        for c, t in table.schema.items()
        if c in scale or t != pl.Float64
    )
