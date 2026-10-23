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


def plot_experience_estimate(pairs: pl.DataFrame) -> Figure:
    """Estimated ``Experience`` on P against Adecco's value, one dot per pair with titles.
    The dashed line is the identity: dots on it are estimated exactly."""
    est, real = "Experience_hat_P", "Experience"
    df = pairs.filter(pl.col(est) > 0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="tab:gray", linewidth=1, zorder=1)
    ax.scatter(
        df[est],
        df[real],
        s=6,
        alpha=0.15,
        color="tab:blue",
        edgecolors="none",
        zorder=2,
    )
    low = min(df.select(pl.col(est).min(), pl.col(real).min()).row(0)) - 0.02
    ax.set_xlim(low, 1.01)
    ax.set_ylim(low, 1.01)
    ax.set_xlabel("estimated Experience on P (cosine similarity)")
    ax.set_ylabel("Adecco Experience")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
