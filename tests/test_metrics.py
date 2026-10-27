import numpy as np
import polars as pl
import pytest

from hiring_cv_bias.fairness.metrics import (
    bootstrap,
    for_display,
    outcome_change_sources,
    parser_changes,
    parser_effect,
    profile_corrections,
    selection_rates,
)
from hiring_cv_bias.fairness.protocol import add_group, add_outcomes


def _outcomes() -> pl.DataFrame:
    # 4 men from the North (reference) and 4 women from the South, outcomes set by hand
    return add_group(
        pl.DataFrame(
            {
                "CANDIDATE_ID": list(range(8)),
                "Gender": ["Male"] * 4 + ["Female"] * 4,
                "Location": ["NORTH"] * 4 + ["SOUTH"] * 4,
                "top_half_adecco": [True, True, False, False] * 2,
                "top_half_O": [True, True, False, False, True, False, False, False],
                "top_half_C": [True, False, True, False, True, True, True, False],
                "w": [1.0] * 8,
            }
        )
    )


def _row(table: pl.DataFrame, group: str) -> dict:
    return table.filter(pl.col("group") == group).to_dicts()[0]


def test_selection_rates_spd_and_di_against_the_reference():
    rates = selection_rates(_outcomes(), ["Gender", "group"])
    female = _row(rates, "Female")
    assert female["rate_O"] == 0.25
    assert female["spd_O"] == pytest.approx(0.25 - 0.5)
    assert female["di_O"] == pytest.approx(0.25 / 0.5)
    assert (_row(rates, "Male")["spd_O"], _row(rates, "Male")["di_O"]) == (0, 1)
    assert _row(rates, "Female_SOUTH")["rate_C"] == 0.75
    assert rates["group"].to_list()[:2] == ["Female", "Male"]


def test_weights_count_a_candidate_several_times():
    df = _outcomes().with_columns(
        pl.when(pl.col("CANDIDATE_ID") == 4).then(3.0).otherwise(1.0).alias("w")
    )
    # candidate 4 (selected with O) now weighs 3: 3 of 6 female pairs are selected
    assert _row(selection_rates(df, ["Gender"]), "Female")["rate_O"] == 0.5


def test_parser_effect_false_negative_and_false_positive_rates():
    effect = parser_effect(_outcomes(), ["Gender"])
    male, female = _row(effect, "Male"), _row(effect, "Female")
    # men: C selects 0 and 2, O selects 0 and 1 → one false negative of two positives,
    # one false positive of two negatives, no rate change
    assert (male["fnr"], male["fpr"], male["rate_change"]) == (0.5, 0.5, 0)
    # women: C selects 4, 5, 6, O only 4 → two false negatives of three, no false
    # positive, rate change +0.5
    assert female["fnr"] == pytest.approx(2 / 3)
    assert female["fpr"] == 0
    assert female["rate_change"] == 0.5
    assert female["fnr_diff"] == pytest.approx(2 / 3 - 0.5)
    assert female["fpr_diff"] == -0.5


def test_parser_changes_share_of_pairs_whose_components_change():
    df = add_group(
        pl.DataFrame(
            {
                "Gender": ["Male", "Male", "Female", "Female"],
                "Location": ["NORTH"] * 4,
                "Card_match_O": [0.0, 1.0, 1.0, 1.0],
                "Card_match_C": [1.0, 1.0, 1.0, 1.0],
                "Languages_match_O": [1.0] * 4,
                "Languages_match_C": [1.0] * 4,
                "Experience_hat_O": [0.9, 0.9, 0.9, 0.9],
                "Experience_hat_C": [0.9, 0.905, 0.95, 0.9],
                "score_O": [3.0, 3.0, 3.0, 3.0],
                "score_C": [3.3, 3.01, 3.1, 3.0],
            }
        )
    )
    changes = parser_changes(df, ["Gender"])
    male, female = _row(changes, "Male"), _row(changes, "Female")
    assert (male["Card_match"], male["Experience"], male["score"]) == (0.5, 0, 1)
    assert (female["Card_match"], female["Experience"], female["score"]) == (
        0,
        0.5,
        0.5,
    )


def _protocol_pairs(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n_jobs, per_job, n_candidates = 40, 6, 80
    n = n_jobs * per_job
    candidate = rng.integers(0, n_candidates, n)
    score = rng.integers(0, 4, n).astype(float)  # few values: many ties
    return add_group(
        pl.DataFrame(
            {
                "Job_id": np.repeat(np.arange(n_jobs), per_job).astype(str),
                "CANDIDATE_ID": candidate,
                "Gender": np.where(candidate % 2 == 0, "Male", "Female"),
                "Location": np.where(candidate % 3 == 0, "SOUTH", "NORTH"),
                "score_adecco": score,
                "score_P": score,
                "score_O": score,
                "score_C": score + (rng.random(n) < 0.2),
            }
        )
    ).unique(["Job_id", "CANDIDATE_ID"])


def test_bootstrap_gives_the_point_estimate_inside_its_interval():
    df = add_outcomes(_protocol_pairs())
    stats = {"rates": selection_rates, "effect": parser_effect}
    out = bootstrap(df, stats, n_boot=40, seed=1)
    point = selection_rates(df)
    rates = out["rates"]
    assert rates["rate_C"].to_list() == point["rate_C"].to_list()
    assert rates["pairs"].to_list() == point["pairs"].to_list()
    assert (
        (rates["rate_C_lo"] <= rates["rate_C"])
        & (rates["rate_C"] <= rates["rate_C_hi"])
    ).all()
    assert {"rate_change_lo", "fnr_diff_lo", "fnr_diff_hi"} <= set(
        out["effect"].columns
    )
    # the same seed gives the same intervals
    again = bootstrap(df, stats, n_boot=40, seed=1)
    assert again["rates"]["rate_C_lo"].to_list() == rates["rate_C_lo"].to_list()


def test_bootstrap_keeps_counts_that_change_between_replicates():
    def flips(d: pl.DataFrame) -> pl.DataFrame:
        # how many pairs change outcome depends on the tie-break of the replicate
        n = d.filter(pl.col("top_half_O") != pl.col("top_half_C")).height
        return pl.DataFrame({"group": ["all"], "flips": [n], "share": [n / d.height]})

    df = add_outcomes(_protocol_pairs())
    out = bootstrap(df, {"flips": flips}, n_boot=40, seed=1)["flips"]
    assert out["flips"].to_list() == flips(df)["flips"].to_list()
    # every replicate counts, not only those with the same number of flips
    assert (out["share_lo"] < out["share_hi"]).all()


def test_for_display_formats_percentages_ratios_and_intervals():
    table = pl.DataFrame(
        {
            "group": ["SOUTH", "NORTH"],
            "pairs": [10, 20],
            "fnr": [0.1017, 0.0749],
            "fnr_diff": [0.0268, -0.0004],
            "fnr_diff_lo": [0.0046, -0.0004],
            "fnr_diff_hi": [0.0536, 0.0],
            "di": [0.9512, 1.0],
        }
    )
    out = for_display(table, ratios=["di"])
    assert out.columns == ["group", "pairs", "fnr", "fnr_diff", "di"]
    assert out["fnr"].to_list() == [10.2, 7.5]
    assert out["fnr_diff"].to_list() == ["2.7 [0.5, 5.4]", "0.0 [0.0, 0.0]"]
    assert out["di"].to_list() == [0.95, 1.0]
    assert for_display(table, ["fnr_diff"]).columns == ["group", "pairs", "fnr_diff"]


def test_profile_corrections_count_candidates_and_items_added():
    o = pl.DataFrame(
        {
            "CANDIDATE_ID": [1, 2, 3],
            "licenses": [[], ["b"], []],
            "languages": [["inglese"], [], []],
            "experience": [[], [], ["cuoco"]],
        }
    )
    c = o.with_columns(
        pl.Series("licenses", [["b"], ["b"], []]),
        pl.Series("languages", [["inglese", "francese", "tedesco"], [], []]),
    )
    out = profile_corrections(o, c, pl.Series([1, 2]))  # candidate 3 is outside
    rows = {r["field"]: r for r in out.to_dicts()}
    assert rows["licenses"]["candidates_corrected"] == 1
    assert rows["licenses"]["share_of_candidates"] == 0.5
    assert rows["languages"]["items_added"] == 2
    assert rows["experience"]["candidates_corrected"] == 0


def test_outcome_change_sources_only_look_at_pairs_that_change():
    df = add_group(
        pl.DataFrame(
            {
                "Gender": ["Male"] * 4,
                "Location": ["NORTH"] * 4,
                "top_half_O": [False, True, True, True],
                "top_half_C": [True, False, False, True],
                "Card_match_O": [0.0, 1.0, 1.0, 0.0],
                "Card_match_C": [1.0, 1.0, 1.0, 1.0],
                "Languages_match_O": [1.0] * 4,
                "Languages_match_C": [1.0] * 4,
                "Experience_hat_O": [0.9] * 4,
                "Experience_hat_C": [0.9, 0.95, 0.9, 0.9],
                "score_O": [3.0, 3.0, 3.0, 3.0],
                "score_C": [3.3, 3.1, 3.0, 3.3],
            }
        )
    )
    row = outcome_change_sources(df, ["Gender"]).to_dicts()[0]
    # three pairs change outcome: one gets a license, one a title, one nothing
    assert row["changed_pairs"] == 3
    assert row["Card_match"] == pytest.approx(1 / 3)
    assert row["Experience"] == pytest.approx(1 / 3)
    assert row["Languages_match"] == 0
    assert row["score_unchanged"] == pytest.approx(1 / 3)
