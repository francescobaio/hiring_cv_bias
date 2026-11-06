import numpy as np
import polars as pl
from fairlearn.metrics import MetricFrame, true_positive_rate

from hiring_cv_bias.fairness.mitigation import (
    TOY_FEATURES,
    apply_equalized_odds,
    coefficients_table,
    decide,
    decision_input,
    equalized_odds_on_decisions,
    equalized_odds_on_score,
    features,
    fit_matcher,
    fit_prejudice_remover,
    flip_table,
    mitigation_table,
    reweighing_weights,
    toy_candidates,
    weights_table,
)


def test_toy_candidates_bias_is_only_in_the_past_decisions() -> None:
    df = toy_candidates(20_000, seed=0)
    assert df.height == 20_000
    by_region = df.group_by("Location").agg(
        pl.col("hired_truth").mean(), pl.col("hired_past").mean()
    )
    north = by_region.filter(pl.col("Location") == "NORTH").row(0, named=True)
    south = by_region.filter(pl.col("Location") == "SOUTH").row(0, named=True)
    assert abs(north["hired_truth"] - south["hired_truth"]) < 0.03
    assert north["hired_past"] == north["hired_truth"]
    # the distance is a proxy of the region: closer in the North, with an overlap
    distance = df.group_by("Location").agg(
        pl.col("distance").min().alias("lo"), pl.col("distance").max().alias("hi")
    )
    assert distance.filter(pl.col("Location") == "NORTH")["hi"][0] <= 0.6
    assert distance.filter(pl.col("Location") == "SOUTH")["lo"][0] >= 0.4
    # 30% of the qualified southern candidates are passed over
    assert abs(south["hired_past"] / south["hired_truth"] - 0.7) < 0.03


def test_features_leave_the_region_out() -> None:
    df = pl.DataFrame(
        {
            "Location": ["NORTH", "SOUTH"],
            "skills": [0.2, 0.8],
            "experience": [0.5, 0.1],
            "distance": [0.1, 0.7],
        }
    )
    x = features(df)
    assert list(x.columns) == list(TOY_FEATURES)
    assert x.to_numpy().tolist() == [[0.2, 0.5, 0.1], [0.8, 0.1, 0.7]]


def test_matcher_learns_the_bias_and_the_prejudice_remover_removes_it() -> None:
    df = toy_candidates(5_000, seed=1)
    plain = fit_matcher(df)
    coef = plain.coef_[0]
    assert coef[0] > 0 and coef[1] > 0  # skills and experience count
    assert coef[2] < -0.5  # living far costs, although the truth ignores it
    removed = fit_prejudice_remover(df, eta=20.0, epochs=10)
    assert abs(removed.coef_[0][2]) < abs(coef[2]) / 2
    table = coefficients_table({"plain": plain, "removed": removed})
    assert table["term"].to_list() == ["intercept", *TOY_FEATURES]
    assert table.columns == ["term", "plain", "removed"]
    decided = decide(df, plain, "plain")
    assert decided["hired_plain"].dtype == pl.Boolean
    # with eta 0 fairlib's training takes the same decisions as scikit-learn's matcher
    same = decide(decided, fit_prejudice_remover(df, eta=0.0), "same")
    assert (same["hired_same"] == same["hired_plain"]).mean() > 0.98


def test_reweighing_weights_match_the_textbook_example() -> None:
    # six rows: three northerners with labels 1, 1, 0 and three southerners with 1, 0, 0
    df = pl.DataFrame(
        {
            "Location": ["NORTH"] * 3 + ["SOUTH"] * 3,
            "skills": [0.5] * 6,
            "experience": [0.5] * 6,
            "distance": [0.5] * 6,
            "hired_past": [True, True, False, True, False, False],
        }
    )
    weights = reweighing_weights(df)
    assert weights.tolist() == [0.75, 0.75, 1.5, 1.5, 0.75, 0.75]
    table = weights_table(df, weights)
    assert table["Location"].to_list() == ["NORTH", "NORTH", "SOUTH", "SOUTH"]
    assert table["hired_past"].to_list() == [True, False, True, False]
    assert table["weight"].to_list() == [0.75, 1.5, 1.5, 0.75]
    assert table["candidates"].to_list() == [2, 1, 1, 2]
    # in the weighted data the label is independent of the group
    weighted = df.with_columns(pl.Series("w", weights))
    rate = weighted.group_by("Location").agg(
        ((pl.col("w") * pl.col("hired_past")).sum() / pl.col("w").sum()).alias("rate")
    )
    assert rate["rate"].n_unique() == 1


def test_equalized_odds_on_decisions_equalizes_both_rates() -> None:
    rng = np.random.default_rng(42)  # not the seed of the draw
    n = 4_000
    south = np.arange(n) % 2 == 1
    truth = rng.random(n) < 0.5
    # the North gets the truth right 90% of the time, the South misses half its positives
    north_decision = np.where(rng.random(n) < 0.9, truth, ~truth)
    south_decision = truth & (rng.random(n) < 0.5)
    df = pl.DataFrame(
        {
            "Location": np.where(south, "SOUTH", "NORTH"),
            "hired_truth": truth,
            "hired_model": np.where(south, south_decision, north_decision),
        }
    )
    optimizer = equalized_odds_on_decisions(df, "model")
    flips = flip_table(optimizer, df, "model")
    assert flips["group"].to_list() == ["NORTH", "SOUTH"]
    assert flips["pairs"].sum() == n
    assert np.ptp(flips["tpr_after"].to_numpy()) < 1e-6
    assert np.ptp(flips["fpr_after"].to_numpy()) < 1e-6
    assert ((flips["keep"] >= 0) & (flips["keep"] <= 1)).all()
    assert ((flips["add"] >= 0) & (flips["add"] <= 1)).all()
    flipped = apply_equalized_odds(df, optimizer, decision_input(df, "model"), "eo")
    after = MetricFrame(
        metrics=true_positive_rate,
        y_true=flipped["hired_truth"].to_numpy(),
        y_pred=flipped["hired_eo"].to_numpy(),
        sensitive_features=flipped["Location"].to_numpy(),
    ).by_group
    assert np.ptp(after.to_numpy()) < 0.05


def test_equalized_odds_levels_the_reference_down_when_the_group_is_inside() -> None:
    # exact rates: the North at (FPR 0.1, TPR 0.9), the South at (0.3, 0.5), which lies
    # inside the North's reachable region. The common point with the fewest errors is
    # the South's own, so the South keeps its decisions and the North is brought down
    def group(region: str, tpr: float, fpr: float, n: int = 1000) -> pl.DataFrame:
        half = n // 2
        return pl.DataFrame(
            {
                "Location": [region] * n,
                "top_half_C": [True] * half + [False] * half,
                "top_half_O": [True] * int(tpr * half)
                + [False] * (half - int(tpr * half))
                + [True] * int(fpr * half)
                + [False] * (half - int(fpr * half)),
            }
        )

    df = pl.concat([group("SOUTH", 0.5, 0.3), group("NORTH", 0.9, 0.1)])
    optimizer = equalized_odds_on_decisions(df, "O", label="C", outcome="top_half")
    flips = flip_table(optimizer, df, "O", label="C", outcome="top_half")
    assert flips["group"].to_list() == ["NORTH", "SOUTH"]
    north, south = flips.row(0, named=True), flips.row(1, named=True)
    assert np.allclose([south["keep"], south["add"]], [1.0, 0.0], atol=1e-6)
    assert north["keep"] < 1
    assert np.allclose(flips["tpr_after"].to_numpy(), 0.5, atol=1e-3)
    assert np.allclose(flips["fpr_after"].to_numpy(), 0.3, atol=1e-3)


def test_equalized_odds_on_score_uses_the_features() -> None:
    train, test = toy_candidates(5_000, seed=3), toy_candidates(5_000, seed=4)
    model = fit_matcher(train)
    optimizer = equalized_odds_on_score(train, model)
    decided = apply_equalized_odds(test, optimizer, features(test), "eo")
    assert decided["hired_eo"].dtype == pl.Boolean
    rates = MetricFrame(
        metrics=true_positive_rate,
        y_true=decided["hired_truth"].to_numpy(),
        y_pred=decided["hired_eo"].to_numpy(),
        sensitive_features=decided["Location"].to_numpy(),
    ).by_group
    assert np.ptp(rates.to_numpy()) < 0.05


def test_mitigation_table_is_perfect_for_the_truth_itself() -> None:
    df = toy_candidates(1_000, seed=2).with_columns(
        pl.col("hired_truth").alias("hired_copy")
    )
    table = mitigation_table(df, ["copy", "past"])
    assert table["method"].to_list() == ["copy", "past"]
    assert table.columns[:6] == [
        "method",
        "accuracy",
        "rate_NORTH",
        "rate_SOUTH",
        "spd",
        "di",
    ]
    copy = table.row(0, named=True)
    assert copy["accuracy"] == 1.0
    assert copy["fnr_NORTH"] == copy["fnr_SOUTH"] == copy["fpr_SOUTH"] == 0.0
    past = table.row(1, named=True)
    assert past["spd"] < 0 and past["di"] < 1
    assert past["fnr_NORTH"] == 0.0 and past["fnr_diff"] > 0
