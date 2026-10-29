from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes
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


# one fixed color per score, the same in every figure; single-series figures use ink
VERSION_COLORS = {"adecco": "#2a78d6", "O": "#eb6834", "C": "#1baf7a"}
VERSION_LABELS = {
    "adecco": "Adecco's score",
    "O": "O (parser)",
    "C": "C (corrected)",
}
INK = "#3d3d3a"


def _group_rows(table: pl.DataFrame) -> dict[str, float]:
    """y position of every group, one block per attribute with a gap between."""
    ys, y = {}, 0.0
    for attribute in table["attribute"].unique(maintain_order=True):
        for group in table.filter(pl.col("attribute") == attribute)["group"]:
            ys[group] = y
            y += 1
        y += 0.6
    return ys


def _dots(ax: Axes, table: pl.DataFrame, col: str, ys: list[float], **style) -> None:
    """One dot per row at ``col`` with its 95% interval, in percent."""
    x = 100 * table[col]
    ax.errorbar(
        x,
        ys,
        xerr=[x - 100 * table[f"{col}_lo"], 100 * table[f"{col}_hi"] - x],
        fmt="o",
        markersize=5,
        elinewidth=1.5,
        zorder=2,
        **style,
    )


def _group_axis(ax: Axes, ys: dict[str, float]) -> None:
    """Group names on the y axis, first group on top."""
    ax.set_yticks(list(ys.values()), list(ys))
    ax.invert_yaxis()


def plot_selection_rates(rates: pl.DataFrame, overall: float) -> Figure:
    """Selection rate of every group with its 95% interval, one color per score. The
    dashed line is the rate of all pairs."""
    ys = _group_rows(rates)
    versions = [v for v in VERSION_COLORS if f"rate_{v}" in rates.columns]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.axvline(100 * overall, color="tab:gray", ls="--", lw=1, zorder=1)
    for i, v in enumerate(versions):
        offset = (i - (len(versions) - 1) / 2) * 0.22
        _dots(
            ax,
            rates,
            f"rate_{v}",
            [ys[g] + offset for g in rates["group"]],
            color=VERSION_COLORS[v],
            label=VERSION_LABELS[v],
        )
    _group_axis(ax, ys)
    ax.set_xlabel("selection rate, % (top half of the job's candidates)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


# the panels of ``plot_parser_effect``: column -> title, one figure per criterion
SEPARATION_PANELS = {
    "rate_change": "selection rate change (C − O)",
    "fnr_diff": "FNR gap from the reference",
    "fpr_diff": "FPR gap from the reference",
}
SUFFICIENCY_PANELS = {
    "ppv_diff": "PPV gap from the reference",
    "npv_diff": "NPV gap from the reference",
}


def plot_parser_effect(
    effect: pl.DataFrame, panels: Mapping[str, str] = SEPARATION_PANELS
) -> Figure:
    """One panel per column of ``panels``, one dot per group with its 95% interval, in
    percentage points: by default the change of the selection rate when the parser is
    corrected and the gaps from the reference group in the false negative and false
    positive rates of O (separation); with ``SUFFICIENCY_PANELS`` the gaps in its
    predictive values (sufficiency)."""
    ys = _group_rows(effect)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.7 * len(panels), 5), sharey=True, squeeze=False
    )
    for ax, (col, title) in zip(axes[0], panels.items(), strict=True):
        ax.axvline(0, color="tab:gray", ls="--", lw=1, zorder=1)
        _dots(ax, effect, col, [ys[g] for g in effect["group"]], color=INK)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("percentage points")
        ax.grid(axis="x", alpha=0.3)
    _group_axis(axes[0][0], ys)
    fig.tight_layout()
    return fig
