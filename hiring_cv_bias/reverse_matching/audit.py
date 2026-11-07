"""Descriptive tables for the data audit of the notebook: list lengths, agreement of the
rules with Adecco's components, comparison of the three profile versions."""

import polars as pl

from hiring_cv_bias.reverse_matching.components import add_components
from hiring_cv_bias.reverse_matching.profiles import PARSER_FIELDS
from hiring_cv_bias.reverse_matching.scoring import RULE_COMPONENTS

# how much a column of the pairs must move between two versions to count as different
TOLERANCES = {c: 0.0 for c in RULE_COMPONENTS} | {"Experience_hat": 0.01, "score": 1e-6}


def differs(col: str, a: str, b: str) -> pl.Expr:
    """Whether ``col`` differs between versions ``a`` and ``b`` by more than its tolerance."""
    return (pl.col(f"{col}_{a}") - pl.col(f"{col}_{b}")).abs() > TOLERANCES[col]


def list_lengths(pairs: pl.DataFrame) -> pl.DataFrame:
    """Number of candidates per job list."""
    n = pairs.group_by("Job_id").len()["len"]
    return pl.DataFrame(
        {
            "jobs": [n.len()],
            "median": [n.median()],
            "mean": [n.mean()],
            "max": [n.max()],
            "jobs_ge_5": [(n >= 5).sum()],
        }
    )


def rules_agreement(
    pairs: pl.DataFrame, profile: pl.DataFrame, jobs_req: pl.DataFrame
) -> pl.DataFrame:
    """Share of pairs where each recomputed component equals Adecco's."""
    df = add_components(pairs, profile, jobs_req, suffix="_rule")
    return df.select(
        [
            ((pl.col(c) - pl.col(c + "_rule")).abs() < 1e-6).mean().alias(c)
            for c in RULE_COMPONENTS
        ]
    ).unpivot(variable_name="component", value_name="agreement")


def profile_field_stats(profiles: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Per version and parser field: mean items per candidate and share of candidates with at
    least one item. ``regulated`` and ``skills`` are left out: they are copied from P, so they are
    the same in every version."""
    return pl.concat(
        prof.select(
            pl.lit(v).alias("version"),
            *[
                e
                for f in PARSER_FIELDS
                for e in (
                    pl.col(f).list.len().mean().alias(f),
                    (pl.col(f).list.len() > 0).mean().alias(f + "_nonempty"),
                )
            ],
        )
        for v, prof in profiles.items()
    )


def profile_overlap(real: pl.DataFrame, original: pl.DataFrame) -> pl.DataFrame:
    """Per parser field: candidates with parsed items, share of them with items in the real
    profile too, and among those the share with a common item and with identical lists."""
    df = real.join(original, on="CANDIDATE_ID", suffix="_O")
    rows = []
    for f in PARSER_FIELDS:
        parsed = df.filter(pl.col(f + "_O").list.len() > 0)
        both = parsed.filter(pl.col(f).list.len() > 0)
        rows.append(
            both.select(
                pl.lit(f).alias("field"),
                pl.lit(parsed.height).alias("candidates_parser"),
                pl.lit(both.height / parsed.height).alias("share_also_in_real"),
                (pl.col(f).list.set_intersection(pl.col(f + "_O")).list.len() > 0)
                .mean()
                .alias("share_common_item"),
                (pl.col(f).list.sort() == pl.col(f + "_O").list.sort())
                .mean()
                .alias("share_identical"),
            )
        )
    return pl.concat(rows)


def version_differences(pairs: pl.DataFrame, a: str, b: str) -> pl.DataFrame:
    """Share of pairs where each component and the score differ between versions ``a`` and ``b``."""
    return (
        pairs.select(differs(c, a, b).mean().alias(c) for c in TOLERANCES)
        .unpivot(variable_name="column", value_name="share_different")
        .with_columns(pl.lit(f"{a} vs {b}").alias("comparison"))
    )
