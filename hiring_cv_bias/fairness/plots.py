from collections.abc import Sequence

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure


def plot_score_distribution(
    scores: Sequence[float], threshold: float, title: str = ""
) -> Figure:
    """Box plot and histogram of the match score, with the outlier threshold."""
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(8, 4.5), sharex=True, height_ratios=[1, 3]
    )
    ax_box.boxplot(scores, vert=False, widths=0.6, flierprops={"markersize": 3})
    ax_box.set_yticks([])
    ax_hist.hist(scores, bins=140, color="tab:blue")
    ax_hist.set_yscale("log")
    ax_hist.set_ylabel("pairs (log scale)")
    ax_hist.set_xlabel("match_score")
    for ax in (ax_box, ax_hist):
        ax.axvline(threshold, color="tab:red", ls="--", lw=1)
    ax_hist.annotate(
        f"upper whisker = {threshold:.2f}",
        (threshold, 0.95),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        color="tab:red",
        va="top",
    )
    ax_box.set_title(title)
    fig.tight_layout()
    return fig


def plot_calibration_curve(pairs: pl.DataFrame) -> Figure:
    """Isotonic calibration curve of ``Experience`` on P against the real values.

    Orange dots: real ``Experience`` averaged in 20 equal-size bins of raw
    similarity; blue steps: the learned curve; dotted line: no calibration.
    """
    raw, cal = "Experience_raw_P", "Experience_hat_P"
    curve = pairs.select(raw, cal).unique().sort(raw)
    bins = (
        pairs.filter(pl.col(raw) > 0)
        .with_columns(pl.col(raw).qcut(20, allow_duplicates=True).alias("bin"))
        .group_by("bin")
        .agg(pl.col(raw).mean(), pl.col("Experience").mean())
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        bins[raw],
        bins["Experience"],
        "o",
        color="tab:orange",
        markersize=6,
        label="real Experience (mean per bin)",
    )
    ax.step(
        curve[raw],
        curve[cal],
        where="post",
        color="tab:blue",
        linewidth=2,
        label="calibration curve",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle=":",
        color="tab:gray",
        linewidth=1,
        label="no calibration (estimate = raw)",
    )
    ax.set_xlabel("raw similarity on P")
    ax.set_ylabel("Experience")
    ax.set_xlim(0.3, 1.01)
    ax.set_ylim(0.3, 1.01)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
