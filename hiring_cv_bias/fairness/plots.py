from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hiring_cv_bias.fairness.robustness import reachable_tpr


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


# one fixed color per region, after the three scores (categorical slots 4 and 5)
REGION_COLORS = {"NORTH": "#eda100", "SOUTH": "#e87ba4"}


def _reachable_hull(fpr: float, tpr: float) -> list[tuple[float, float]]:
    """The rates a group with point (fpr, tpr) can reach by flipping its decisions at
    random: the quadrilateral of (0, 0), its point, (1, 1) and the mirrored point."""
    return [(0.0, 0.0), (fpr, tpr), (1.0, 1.0), (1 - fpr, 1 - tpr)]


def plot_reachable_region(
    effect: pl.DataFrame,
    reference: str = "NORTH",
    group: str = "SOUTH",
    attribute: str = "Location",
) -> Figure:
    """The leveling-down argument of a post-processing that flips decisions at random
    per group (Hardt et al. 2016), in ROC space (FPR, TPR). Each group's region holds the
    rates it can reach; equalized odds needs a point inside both. Left: the whole space;
    right: a zoom around the two points, with the TPR the reference could reach at the
    group's FPR (dotted). When the group's point lies inside the reference's region, the
    best common point is the group's own."""
    r = reachable_tpr(effect, attribute, reference, group).row(0, named=True)
    fpr_r, tpr_r = r["fpr_reference"], r["tpr_reference"]
    fpr_g, tpr_g = r["fpr_group"], r["tpr_group"]
    reachable = r["reachable_tpr_at_group_fpr"]
    points = {reference: (fpr_r, tpr_r), group: (fpr_g, tpr_g)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    lo_x, hi_x = 0.0, max(fpr_r, fpr_g) + 0.08
    lo_y, hi_y = min(tpr_r, tpr_g) - 0.08, 1.0
    for ax, zoom in zip(axes, (False, True), strict=True):
        ax.plot([0, 1], [0, 1], color="tab:gray", ls="--", lw=1, zorder=1)
        for g, (fpr, tpr) in points.items():
            color = REGION_COLORS.get(g, INK)
            ax.add_patch(
                plt.Polygon(
                    _reachable_hull(fpr, tpr),
                    closed=True,
                    facecolor=color,
                    alpha=0.18,
                    edgecolor=color,
                    lw=2,
                    zorder=2,
                )
            )
            ax.plot(
                fpr,
                tpr,
                "o",
                markersize=8,
                color=color,
                markeredgecolor="white",
                markeredgewidth=2,
                zorder=4,
            )
            # in the zoom the reference's label goes up-left, clear of the dotted line
            left = zoom and g == reference
            ax.annotate(
                f"{g.title()} ({fpr:.3f}, {tpr:.3f})" if zoom else g.title(),
                (fpr, tpr),
                xytext=(-6, 8) if left else (8, -4 if g == group else 6),
                textcoords="offset points",
                ha="right" if left else "left",
                color=INK,
                fontsize=9,
                zorder=5,
            )
        if zoom:
            ax.plot(
                [fpr_g, fpr_g], [tpr_g, reachable], color=INK, ls=":", lw=1.5, zorder=3
            )
            ax.plot(fpr_g, reachable, "o", markersize=5, color=INK, zorder=4)
            ax.annotate(
                f"{reference.title()} reachable at the same FPR: {reachable:.3f}",
                (fpr_g, reachable),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                color=INK,
                fontsize=9,
            )
            ax.set_xlim(lo_x, hi_x)
            ax.set_ylim(lo_y, hi_y)
            ax.set_title("zoom around the two points", fontsize=10)
        else:
            ax.add_patch(
                plt.Rectangle(
                    (lo_x, lo_y),
                    hi_x - lo_x,
                    hi_y - lo_y,
                    fill=False,
                    edgecolor=INK,
                    lw=1,
                    ls="-",
                    zorder=3,
                )
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_title("ROC space; dashed: TPR = FPR", fontsize=10)
        ax.set_xlabel("false positive rate (FPR)")
        ax.set_ylabel("true positive rate (TPR = 1 − FNR)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# the classes of ``list_class``, as short labels for a figure
CLASS_LABELS = {
    "none required": "no skill required",
    "required, same for all": "skills required,\nsame for all",
    "required, discriminate": "skills that separate\nthe candidates",
}


def plot_error_by_class(
    by_class: pl.DataFrame,
    reference: str = "NORTH",
    group: str = "SOUTH",
    labels: Mapping[str, str] = CLASS_LABELS,
) -> Figure:
    """The error rate of ``reference`` and ``group`` in every class of lists, with their
    95% intervals, from a bootstrapped ``error_by_class`` table; the gap of ``group``
    with its interval is written on the right of each row."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ys = {c: i for i, c in enumerate(by_class["class"])}
    for offset, region in ((-0.14, reference), (0.14, group)):
        col = f"error_{region.lower()}"
        _dots(
            ax,
            by_class,
            col,
            [ys[c] + offset for c in by_class["class"]],
            color=REGION_COLORS.get(region, INK),
            label=region.title(),
        )
    for r in by_class.iter_rows(named=True):
        ax.annotate(
            f"gap {100 * r['gap']:+.1f} [{100 * r['gap_lo']:.1f}, {100 * r['gap_hi']:.1f}]",
            (1.02, ys[r["class"]]),
            xycoords=("axes fraction", "data"),
            va="center",
            color=INK,
            fontsize=9,
        )
    ax.set_yticks(list(ys.values()), [labels.get(c, c) for c in ys])
    ax.invert_yaxis()
    ax.set_xlabel("error rate, % (outcome differs between O and C)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    return fig
