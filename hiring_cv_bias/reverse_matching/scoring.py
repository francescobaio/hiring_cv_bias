"""The reverse-matching score as a weighted sum of its seven components, recovered by
least squares after removing the score outliers (section 1.2 of the notebook)."""

from collections.abc import Mapping

import numpy as np
import polars as pl

COMPONENTS = [
    "Experience",
    "Interest",
    "Card_match",
    "Languages_match",
    "Regulated_Trainings_match",
    "match_skills",
    "Distance_match",
]
# the components we can recompute from a profile and the job requirements
RULE_COMPONENTS = [c for c in COMPONENTS if c not in ("Experience", "Interest")]


def score_outliers(pairs: pl.DataFrame) -> pl.Series:
    """Scores above the upper whisker of the box plot: Q3 + 1.5 · (Q3 − Q1)."""
    q1, q3 = pairs.select(
        pl.col("match_score").quantile(0.25).alias("q1"),
        pl.col("match_score").quantile(0.75).alias("q3"),
    ).row(0)
    return (pairs["match_score"] > q3 + 1.5 * (q3 - q1)).alias("outlier")


def score_expr(
    weights: Mapping[str, float], columns: Mapping[str, str] | None = None
) -> pl.Expr:
    """The formula as a polars expression; ``columns`` renames the component columns."""
    columns = columns or {}
    expr: pl.Expr = pl.lit(weights["intercept"])
    for c in COMPONENTS:
        expr = expr + weights[c] * pl.col(columns.get(c, c))
    return expr


def validate_formula(pairs: pl.DataFrame, weights: Mapping[str, float]) -> pl.DataFrame:
    """Share of scores the formula reproduces within 1e-4, and where the others are."""
    df = pairs.with_columns(
        (pl.col("match_score") - score_expr(weights)).alias("residual")
    )
    off = df.filter(pl.col("residual").abs() > 1e-4)
    return pl.DataFrame(
        {
            "n_pairs": [df.height],
            "share_exact": [1 - off.height / df.height],
            "n_pairs_off": [off.height],
            "n_jobs_off": [off["Job_id"].n_unique()],
            "residual_min": [off["residual"].min()],
            "residual_max": [off["residual"].max()],
        }
    )


def fit_weights(pairs: pl.DataFrame) -> dict[str, float]:
    """Least-squares weights of the components (and intercept) on the given pairs."""
    X = np.hstack([pairs.select(COMPONENTS).to_numpy(), np.ones((pairs.height, 1))])
    y = pairs["match_score"].to_numpy()
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(zip(COMPONENTS + ["intercept"], (float(v) for v in w), strict=True))
