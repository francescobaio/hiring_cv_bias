import numpy as np
import polars as pl

from hiring_cv_bias.fairness.metrics import parser_effect
from hiring_cv_bias.fairness.protocol import add_group, add_outcomes
from hiring_cv_bias.fairness.robustness import (
    LIST_CLASSES,
    MARGIN_BANDS,
    error_by_class,
    error_by_margin_band,
    list_class,
    list_classes_table,
    margin_band,
    margin_table,
    margin_to_cutoff,
    perturbation_flips,
    reachable_tpr,
    standardized_error,
)
from hiring_cv_bias.reverse_matching.scoring import RULE_COMPONENTS


def lists() -> pl.DataFrame:
    """Four lists of four: j1 and j3 northern, j2 and j4 southern; j1 and j2 require
    skills that separate the candidates, j3 and j4 require none. Scores 4, 3, 2, 1, so
    the cutoff score is 3 everywhere; C differs from O in one pair of j2 and j4."""
    df = add_group(
        pl.DataFrame(
            {
                "Job_id": [j for j in ("j1", "j2", "j3", "j4") for _ in range(4)],
                "CANDIDATE_ID": list(range(16)),
                "Gender": ["Female"] * 16,
                "Location": (["NORTH"] * 4 + ["SOUTH"] * 4) * 2,
                "score_adecco": [4.0, 3.0, 3.0, 1.0] * 4,
                "score_O": [4.0, 3.0, 2.0, 1.0] * 4,
                "score_C": [4.0, 3.0, 2.0, 1.0, 4.0, 2.0, 3.0, 1.0] * 2,
                "match_skills_O": [1.0, 0.5, 0.0, 0.0] * 2 + [0.0] * 8,
                "skills_required": [True] * 8 + [False] * 8,
                "Interest": [1.0] * 16,
                "Experience_hat_O": [0.9, 0.85, 0.8, 0.5] * 4,
            }
        )
    )
    for c in RULE_COMPONENTS:
        if f"{c}_O" not in df.columns:
            df = df.with_columns(pl.lit(1.0).alias(f"{c}_O"))
    return add_outcomes(df, ["score_adecco", "score_O", "score_C"])


def test_margin_and_bands():
    df = lists()
    margin = df.select(margin_to_cutoff("O").alias("m"))["m"]
    assert margin.min() == -2.0 and margin.max() == 1.0 and (margin == 0).sum() == 4
    bands = df.select(margin_band(margin_to_cutoff("O")).alias("b"))["b"].to_list()
    assert set(bands) == {MARGIN_BANDS[0], MARGIN_BANDS[4]}


def test_margin_table_counts_ties_and_close_pairs():
    table = margin_table(lists(), ["Location"])
    assert table["group"].to_list() == ["NORTH", "SOUTH"]
    # one tied pair (margin 0) in each list of four: within 0.1 is one in four
    assert table["within_0.1"].to_list() == [0.25, 0.25]
    assert table["draw_O"].to_list() == [0.0, 0.0]
    # Adecco's score ties the second and third of every list across the cut
    assert table["draw_adecco"].to_list() == [0.5, 0.5]


def test_error_by_band_and_standardization():
    df = lists()
    by_band = error_by_margin_band(df)
    assert by_band["band"].to_list() == [MARGIN_BANDS[0]] * 2 + [MARGIN_BANDS[4]] * 2
    # the South's two errors sit in the pairs with margins 0 and 1 of j2 and j4
    std = standardized_error(df)
    south = std.filter(pl.col("group") == "SOUTH").row(0, named=True)
    assert south["error"] == 0.5 and south["gap"] == 0.5
    # same distribution of bands in both regions: standardized equals crude
    assert abs(south["error_std"] - south["error"]) < 1e-12
    assert std.filter(pl.col("group") == "NORTH")["gap_std"][0] == 0.0


def test_list_classes_and_gaps():
    df = lists()
    classes = list_class(pl.col("skills_required"))
    table = list_classes_table(df, classes)
    assert table.select("Location", "class", "lists", "pairs").rows() == [
        ("NORTH", LIST_CLASSES[0], 1, 4),
        ("NORTH", LIST_CLASSES[2], 1, 4),
        ("SOUTH", LIST_CLASSES[0], 1, 4),
        ("SOUTH", LIST_CLASSES[2], 1, 4),
    ]
    assert table["share_of_lists"].to_list() == [0.5] * 4
    errors = error_by_class(df, classes)
    # the class nobody is in is left out; in both classes the South's second and third
    # pairs swap with C, the North's never do
    assert errors.columns == ["class", "pairs", "error_north", "error_south", "gap"]
    assert errors["class"].to_list() == [LIST_CLASSES[0], LIST_CLASSES[2]]
    assert errors["pairs"].to_list() == [4, 4]
    assert errors["error_north"].to_list() == [0.0, 0.0]
    assert errors["gap"].to_list() == [0.5, 0.5]


def test_perturbation_flips_nothing_without_noise_and_something_with_it():
    df = lists()
    quiet = perturbation_flips(df, sigmas=(0.0,), n_draws=2)
    assert quiet["flipped"].to_list() == [0.0, 0.0]
    loud = perturbation_flips(df, sigmas=(5.0,), n_draws=20, seed=1)
    assert loud["flipped"].min() > 0
    assert loud.columns == ["sigma", "Location", "pairs", "flipped"]


def test_reachable_tpr_inside_and_outside():
    effect = parser_effect(lists(), ["Location"])
    inside = reachable_tpr(effect)
    # North: TPR 1, FPR 0; South: TPR 0.5, FPR 0.5. From (0, 1) to (1, 1) the North
    # reaches TPR 1 at the South's FPR, so the South's point is inside
    assert inside["reachable_tpr_at_group_fpr"][0] == 1.0
    assert bool(inside["group_inside"][0])
    swapped = reachable_tpr(effect, reference="SOUTH", group="NORTH")
    # the South, from (0.5, 0.5) toward (0, 0), reaches TPR 0 at FPR 0: the North is out
    assert swapped["reachable_tpr_at_group_fpr"][0] == 0.0
    assert not bool(swapped["group_inside"][0])
    assert np.isclose(swapped["tpr_group"][0], 1.0)
