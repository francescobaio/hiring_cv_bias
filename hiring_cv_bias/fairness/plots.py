from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_score_distribution(
    scores: Sequence[float], threshold: float, title: str = ""
) -> Figure:
    """Box plot and histogram of the match score, with the outlier threshold (the upper
    whisker) and the outlier region shaded. The outliers of the box plot are drawn as
    small transparent dots with a random vertical offset, so that they do not hide each
    other."""
    values = np.asarray(scores, dtype=float)
    n_out = int((values > threshold).sum())
    lo, hi = np.floor(values.min()), np.ceil(values.max())

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(8, 4.4), sharex=True, height_ratios=[1.4, 5]
    )
    ax_box.boxplot(
        values,
        orientation="horizontal",
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": BAND, "edgecolor": INK, "linewidth": 1},
        whiskerprops={"color": INK, "linewidth": 1},
        capprops={"color": INK, "linewidth": 1},
        medianprops={"color": VERSION_COLORS["adecco"], "linewidth": 2},
    )
    outliers = values[values > threshold]
    jitter = np.random.default_rng(0).uniform(-0.22, 0.22, outliers.size)
    ax_box.scatter(
        outliers,
        1 + jitter,
        s=6,
        color=VERSION_COLORS["adecco"],
        alpha=0.35,
        linewidths=0,
        zorder=3,
    )
    ax_box.set_ylim(0.6, 1.4)
    ax_box.axis("off")

    ax_hist.hist(
        values,
        bins=np.arange(lo, hi + 0.25, 0.25),
        color=VERSION_COLORS["adecco"],
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    ax_hist.set_yscale("log")
    ax_hist.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax_hist.set_ylabel("pairs (log scale)", color=INK)
    ax_hist.set_xlabel("match_score", color=INK)
    ax_hist.set_xlim(lo - 0.5, hi)
    ax_hist.set_ylim(bottom=0.7)
    for side in ("top", "right", "left"):
        ax_hist.spines[side].set_visible(False)
    ax_hist.spines["bottom"].set_color(MUTED)
    ax_hist.tick_params(axis="y", which="both", length=0, colors=INK)
    ax_hist.tick_params(axis="x", colors=INK)
    ax_hist.grid(axis="y", color=BAND, lw=1)
    ax_hist.set_axisbelow(True)

    for ax in (ax_box, ax_hist):
        ax.axvspan(threshold, hi, color=BAND, lw=0, zorder=0)
        ax.axvline(threshold, color=MUTED, ls="--", lw=1, zorder=1)
    ax_hist.annotate(
        f"upper whisker = {threshold:.2f}",
        (threshold, 1),
        xycoords=("data", "axes fraction"),
        xytext=(5, -2),
        textcoords="offset points",
        color=INK,
        fontsize=9,
        va="top",
    )
    ax_hist.annotate(
        f"outliers: {n_out:,} pairs ({n_out / len(values):.1%})",
        ((threshold + hi) / 2, 1),
        xycoords=("data", "axes fraction"),
        xytext=(0, -2),
        textcoords="offset points",
        color=INK,
        fontsize=9,
        ha="center",
        va="top",
    )
    if title:
        ax_box.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    return fig


def plot_experience_estimate(estimate: np.ndarray, adecco: np.ndarray) -> Figure:
    """``estimate`` of ``Experience`` against Adecco's value, one dot per pair with titles
    (estimate above 0). The dashed line is the identity: dots on it are estimated exactly."""
    est, real = "estimate", "adecco"
    df = pl.DataFrame({est: estimate, real: adecco}).filter(pl.col(est) > 0)
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
MUTED = "#8a8984"
BAND = "#f1f0ec"

# the title of each block of groups, written above it
ATTRIBUTE_TITLES = {
    "Gender": "Gender",
    "Location": "Location",
    "group": "Gender × location",
}


def _group_label(group: str) -> str:
    """``Female_NORTH`` as ``Female, North`` and ``NORTH`` as ``North``."""
    return ", ".join(part.title() for part in group.split("_"))


def _group_rows(table: pl.DataFrame) -> dict[str, float]:
    """y position of every group, one block per attribute with a gap between for its
    title."""
    ys, y = {}, 0.0
    for attribute in table["attribute"].unique(maintain_order=True):
        for group in table.filter(pl.col("attribute") == attribute)["group"]:
            ys[group] = y
            y += 1
        y += 0.9
    return ys


def _dots(ax: Axes, table: pl.DataFrame, col: str, ys: list[float], **style) -> None:
    """One dot per row at ``col``, in percent, with its 95% interval when the table has
    the columns ``<col>_lo`` and ``<col>_hi``. A white ring keeps close dots apart."""
    x = 100 * table[col]
    has_interval = f"{col}_lo" in table.columns
    ax.errorbar(
        x,
        ys,
        xerr=[x - 100 * table[f"{col}_lo"], 100 * table[f"{col}_hi"] - x]
        if has_interval
        else None,
        fmt="o",
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=1,
        elinewidth=1.4,
        zorder=3,
        **style,
    )


def _clean_axis(ax: Axes) -> None:
    """Only the bottom spine, a light vertical grid, text in ink."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.tick_params(axis="x", colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.grid(axis="x", color=BAND, lw=1)
    ax.set_axisbelow(True)


def _bands(ax: Axes, ys: dict[str, float]) -> None:
    """A light band behind every other row, so the dots of one row read together."""
    for i, y in enumerate(ys.values()):
        if i % 2 == 0:
            ax.axhspan(y - 0.45, y + 0.45, color=BAND, lw=0, zorder=0)


def _group_axis(
    ax: Axes, ys: dict[str, float], table: pl.DataFrame | None = None
) -> None:
    """Row names on the y axis, first row on top. With a ``table`` of groups (one
    ``attribute`` column), the names are written as ``Female, North`` and each block of
    the same attribute gets its title above it."""
    grouped = table is not None and "attribute" in table.columns
    ax.set_yticks(list(ys.values()), [_group_label(g) if grouped else g for g in ys])
    if table is not None and grouped:
        for attribute in table["attribute"].unique(maintain_order=True):
            first = table.filter(pl.col("attribute") == attribute)["group"][0]
            ax.annotate(
                ATTRIBUTE_TITLES.get(attribute, attribute),
                (0, ys[first] - 0.9),
                xycoords=("axes fraction", "data"),
                xytext=(-8, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontweight="bold",
                color=INK,
            )
    top = min(ys.values()) - (1.2 if grouped else 0.6)
    ax.set_ylim(max(ys.values()) + 0.6, top)


def plot_selection_rates(rates: pl.DataFrame, overall: float) -> Figure:
    """Selection rate of every group with its 95% interval, one color per score. The
    dashed line is the rate of all pairs."""
    ys = _group_rows(rates)
    versions = [v for v in VERSION_COLORS if f"rate_{v}" in rates.columns]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    _clean_axis(ax)
    ax.axvline(100 * overall, color=MUTED, ls="--", lw=1, zorder=1)
    for i, v in enumerate(versions):
        offset = (i - (len(versions) - 1) / 2) * 0.24
        _dots(
            ax,
            rates,
            f"rate_{v}",
            [ys[g] + offset for g in rates["group"]],
            color=VERSION_COLORS[v],
            label=VERSION_LABELS[v],
        )
    _bands(ax, ys)
    _group_axis(ax, ys, rates)
    ax.annotate(
        f"all pairs {100 * overall:.1f}%",
        (100 * overall, 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -4),
        textcoords="offset points",
        va="top",
        color=MUTED,
        fontsize=9,
    )
    ax.set_xlabel("selection rate, % (top half of the job's candidates)")
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(versions),
        labelcolor=INK,
        handletextpad=0.3,
        columnspacing=1.5,
    )
    fig.tight_layout()
    return fig


# the panels of ``plot_parser_effect``: column -> title, one figure per criterion
SEPARATION_PANELS = {
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
    percentage points: by default the gaps from the reference group in the false negative
    and false positive rates of O (separation); with ``SUFFICIENCY_PANELS`` the gaps in
    its predictive values (sufficiency)."""
    ys = _group_rows(effect)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.7 * len(panels), 5), sharey=True, squeeze=False
    )
    for ax, (col, title) in zip(axes[0], panels.items(), strict=True):
        _clean_axis(ax)
        _bands(ax, ys)
        ax.axvline(0, color=MUTED, ls="--", lw=1, zorder=1)
        _dots(ax, effect, col, [ys[g] for g in effect["group"]], color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("percentage points")
    _group_axis(axes[0][0], ys, effect)
    fig.tight_layout()
    return fig


# one fixed color per region, after the three scores (categorical slots 4 and 5)
REGION_COLORS = {"NORTH": "#eda100", "SOUTH": "#e87ba4"}


def _reachable_hull(fpr: float, tpr: float) -> list[tuple[float, float]]:
    """The rates a group with point (fpr, tpr) can reach by flipping its decisions at
    random: the quadrilateral of (0, 0), its point, (1, 1) and the mirrored point."""
    return [(0.0, 0.0), (fpr, tpr), (1.0, 1.0), (1 - fpr, 1 - tpr)]


def plot_reachable_region(
    flips: pl.DataFrame, reference: str = "NORTH", group: str = "SOUTH"
) -> Figure:
    """The leveling-down argument of a post-processing that flips decisions at random
    per group (Hardt et al. 2016), in ROC space (FPR, TPR), from the table of
    ``flip_table``. Each group's region holds the rates it can reach; equalized odds
    needs a point inside both, and the optimizer picks the one with the fewest errors.
    Left: the whole space; right: a zoom around the two points, with the common point
    after the flips (black) and the move of each group to it (dotted). When the group's
    point lies inside the reference's region, the common point is the group's own and
    only the reference moves."""
    rows = {r["group"]: r for r in flips.iter_rows(named=True)}
    fpr_r, tpr_r = rows[reference]["fpr"], rows[reference]["tpr"]
    fpr_g, tpr_g = rows[group]["fpr"], rows[group]["tpr"]
    fpr_after, tpr_after = rows[group]["fpr_after"], rows[group]["tpr_after"]
    points = {reference: (fpr_r, tpr_r), group: (fpr_g, tpr_g)}
    group_moves = abs(fpr_after - fpr_g) + abs(tpr_after - tpr_g) > 1e-3
    moves = [(fpr_r, tpr_r), (fpr_g, tpr_g)] if group_moves else [(fpr_r, tpr_r)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    lo_x, hi_x = 0.0, max(fpr_r, fpr_g, fpr_after) + 0.08
    lo_y, hi_y = min(tpr_r, tpr_g, tpr_after) - 0.08, 1.0
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
            for fpr, tpr in moves:
                ax.plot(
                    [fpr, fpr_after],
                    [tpr, tpr_after],
                    color=INK,
                    ls=":",
                    lw=1.5,
                    zorder=3,
                )
            ax.plot(fpr_after, tpr_after, "o", markersize=5, color=INK, zorder=4)
            ax.annotate(
                f"after the flips: ({fpr_after:.3f}, {tpr_after:.3f})",
                (fpr_after, tpr_after),
                # below the group's own label when the two points are close
                xytext=(8, -28 if group_moves else -14),
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


MITIGATION_PANELS = {
    "rate": "selection rate, %",
    "fnr": "false negative rate, %",
    "fpr": "false positive rate, %",
}


def plot_mitigation(
    table: pl.DataFrame,
    regions: Sequence[str] = ("NORTH", "SOUTH"),
    panels: Mapping[str, str] = MITIGATION_PANELS,
) -> Figure:
    """The selection rate and the two error rates of every region, one row per method of
    a ``mitigation_table``; the gap between the two dots of a row is what each
    mitigation closes."""
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.4 * len(panels), 3), sharey=True
    )
    ys = {m: float(i) for i, m in enumerate(table["method"])}
    for ax, (rate, title) in zip(axes, panels.items(), strict=True):
        for region in regions:
            _dots(
                ax,
                table,
                f"{rate}_{region}",
                list(ys.values()),
                color=REGION_COLORS.get(region, INK),
                label=region.title(),
            )
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.3)
    _group_axis(axes[0], ys)
    axes[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    return fig
