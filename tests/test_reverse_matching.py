import numpy as np
import polars as pl
import pytest

from hiring_cv_bias.reverse_matching.components import add_components, job_requirements
from hiring_cv_bias.reverse_matching.experience import calibration_table
from hiring_cv_bias.reverse_matching.loader import dedupe_pairs
from hiring_cv_bias.reverse_matching.profiles import (
    build_profile_corrected,
    build_profile_original,
    build_profile_real,
    normalize_item,
    split_items,
)
from hiring_cv_bias.reverse_matching.scoring import (
    fit_weights,
    score_expr,
    score_outliers,
    validate_formula,
)

# the formula recovered from the Adecco data, used as known truth in the tests
WEIGHTS = {
    "Experience": 2.0,
    "Interest": 1.0,
    "Card_match": 0.3,
    "Languages_match": 0.3,
    "Regulated_Trainings_match": 0.3,
    "match_skills": 1.0,
    "Distance_match": 1.0,
    "intercept": 0.0,
}


def score_of(components: dict[str, float]) -> float:
    return pl.DataFrame([components]).select(score_expr(WEIGHTS)).item()


def test_normalize_item_removes_gender_suffix_and_case():
    s = pl.Series(["Impiegato (m/f)", "  Warehouse Worker (other) ", None])
    assert s.to_frame("x").select(normalize_item(pl.col("x")))["x"].to_list() == [
        "impiegato",
        "warehouse worker",
        "",
    ]


def test_split_items_dedupes_and_drops_empty():
    s = pl.Series(["B;A;B; ;b", None])
    assert s.to_frame("x").select(split_items(pl.col("x")))["x"].to_list() == [
        ["b", "a"],
        [],
    ]


def test_match_score_ord_matches_hand_calculation():
    comps = {
        "Experience": 0.9477471,
        "Interest": 0.0,
        "Card_match": 1.0,
        "Languages_match": 1.0,
        "Regulated_Trainings_match": 1.0,
        "match_skills": 0.0,
        "Distance_match": 0.6,
    }
    # first row of ReverseMatching.xlsx: score 3.3954942
    assert score_of(comps) == pytest.approx(3.3954942, abs=1e-6)


def test_match_score_all_components_one():
    comps = dict.fromkeys(WEIGHTS, 1.0)
    comps.pop("intercept")
    # 2 + 1 + 3*0.3 + 1 + 1
    assert score_of(comps) == pytest.approx(5.9)


def _pairs_frame():
    return pl.DataFrame(
        {
            "Job_id": ["ORD_1", "ORD_1", "EXP_2"],
            "CANDIDATE_ID": [1, 2, 1],
            "Experience": [1.0, 0.5, 1.0],
            "Interest": [0.0, 1.0, 0.0],
            "Card_match": [1.0, 0.0, 1.0],
            "Languages_match": [1.0, 1.0, 1.0],
            "Regulated_Trainings_match": [1.0, 1.0, 1.0],
            "match_skills": [0.0, 0.5, 0.0],
            "Distance_match": [1.0, 0.6, 0.2],
            "match_score": [3.9, 3.7 + 14.0, 2.0 + 0.675 + 0.2 + 0.225],
        }
    )


def test_validate_formula_counts_pairs_off_formula():
    summary = validate_formula(_pairs_frame(), WEIGHTS).to_dicts()[0]
    assert summary["share_exact"] == pytest.approx(2 / 3)
    assert summary["n_pairs_off"] == 1 and summary["n_jobs_off"] == 1
    assert summary["residual_max"] == pytest.approx(14.0)


def test_fit_weights_recovers_known_weights():
    rng = np.random.default_rng(0)
    n = 500
    comps = {
        "Experience": rng.uniform(0.8, 1.0, n),
        "Interest": rng.integers(0, 2, n).astype(float),
        "Card_match": rng.integers(0, 2, n).astype(float),
        "Languages_match": rng.integers(0, 2, n).astype(float),
        "Regulated_Trainings_match": rng.integers(0, 2, n).astype(float),
        "match_skills": rng.uniform(0, 1, n),
        "Distance_match": rng.choice([1.0, 0.6, 0.2], n),
    }
    df = pl.DataFrame(comps).with_columns(score_expr(WEIGHTS).alias("match_score"))
    w = fit_weights(df)
    for k, v in WEIGHTS.items():
        assert w[k] == pytest.approx(v, abs=1e-6)
    noisy = df.with_columns(pl.col("match_score") + rng.normal(0, 0.01, n))
    assert {k: round(v, 1) for k, v in fit_weights(noisy).items()} == WEIGHTS


def test_score_outliers_flags_scores_above_the_upper_whisker():
    # quartiles 3 and 6, whisker at 6 + 1.5 · 3 = 10.5
    df = pl.DataFrame({"match_score": [3.0, 3, 3, 3, 6, 6, 6, 6, 30]})
    assert score_outliers(df).to_list() == [False] * 8 + [True]


def test_fit_weights_after_dropping_score_outliers():
    rng = np.random.default_rng(1)
    n = 400
    df = pl.DataFrame(
        {c: rng.uniform(0, 1, n) for c in WEIGHTS if c != "intercept"}
    ).with_columns(score_expr(WEIGHTS).alias("match_score"))
    bonus = np.where(np.arange(n) < 10, 25.0, 0.0)
    noisy = df.with_columns(pl.col("match_score") + bonus)
    w = fit_weights(noisy.filter(~score_outliers(noisy)))
    for k, v in WEIGHTS.items():
        assert w[k] == pytest.approx(v, abs=1e-6)
    assert fit_weights(noisy)["intercept"] > 0.1


def test_dedupe_pairs_keeps_first():
    df = pl.DataFrame(
        {"CANDIDATE_ID": [1, 1, 2], "Job_id": ["a", "a", "a"], "x": [1, 2, 3]}
    )
    assert dedupe_pairs(df)["x"].to_list() == [1, 3]


def _candidates():
    return pl.DataFrame(
        {
            "CANDIDATE_ID": [1, 2, 3],
            "matterknowledges": ["Excel;Word", None, "Picking"],
            "languages": ["Inglese;Italiano", None, "Francese"],
            "regulatedtrainings": [None, "Ragioneria", None],
            "drivinglicenses": ["B", None, "B;A"],
            "experience": ["cassiere;commesso", None, "muratore"],
        }
    )


def _parser():
    return pl.DataFrame(
        {
            "CANDIDATE_ID": [1, 1, 1, 3, 3],
            "Skill": [
                "Driver License B",
                "English",
                "Cashier (m/f)",
                "French",
                "Bricklayer (m/f)",
            ],
            "Skill_Type": [
                "DRIVERSLIC",
                "Language_Skill",
                "Job_title",
                "Language_Skill",
                "Job_title",
            ],
            "Skill_it": [None, None, "Cassiere (m/f)", None, "Muratore (m/f)"],
        }
    )


def _jobs():
    return pl.DataFrame(
        {
            "Job_id": ["ORD_1", "ORD_2"],
            "professional_category": ["Cassiere (m/f)", "Muratore"],
            "matterknowledge": ["Excel;Word;Picking", None],
            "regulatedtraining": [None, None],
            "drivinglicense": ["B", "B;A"],
            "language": ["Inglese", None],
        }
    )


def _profile_original():
    return build_profile_original(build_profile_real(_candidates()), _parser())


def test_profile_real_lists():
    p = build_profile_real(_candidates())
    row = p.filter(pl.col("CANDIDATE_ID") == 1).to_dicts()[0]
    assert row["licenses"] == ["b"]
    assert row["languages"] == ["inglese", "italiano"]
    assert row["skills"] == ["excel", "word"]
    assert row["regulated"] == []
    assert p.filter(pl.col("CANDIDATE_ID") == 2).to_dicts()[0]["licenses"] == []


def test_profile_original_uses_parser_and_keeps_skills():
    o = _profile_original()
    r1 = o.filter(pl.col("CANDIDATE_ID") == 1).to_dicts()[0]
    assert r1["licenses"] == ["b"]
    assert r1["languages"] == ["inglese"]  # Italian was not parsed
    assert r1["experience"] == ["cassiere"]
    assert r1["skills"] == ["excel", "word"]  # copied from P
    r2 = o.filter(pl.col("CANDIDATE_ID") == 2).to_dicts()[0]
    assert r2["licenses"] == [] and r2["languages"] == [] and r2["experience"] == []
    r3 = o.filter(pl.col("CANDIDATE_ID") == 3).to_dicts()[0]
    assert r3["licenses"] == []  # parser missed both licenses


def test_profile_corrected_adds_false_negatives_only_in_whitelist():
    fn_lic = pl.DataFrame({"CANDIDATE_ID": [3], "skill": ["driver_license"]})
    fn_lang = pl.DataFrame({"CANDIDATE_ID": [1, 1], "skill": ["it", "xx"]})
    fn_job = pl.DataFrame(
        {"CANDIDATE_ID": [1, 1], "skill": ["sales assistant", "medium"]}
    )
    c = build_profile_corrected(_profile_original(), fn_lic, fn_lang, fn_job)
    r1 = c.filter(pl.col("CANDIDATE_ID") == 1).to_dicts()[0]
    assert r1["languages"] == ["inglese", "italiano"]
    assert r1["experience"] == ["cassiere", "commesso"]  # "medium" ignored
    r3 = c.filter(pl.col("CANDIDATE_ID") == 3).to_dicts()[0]
    assert r3["licenses"] == ["b"]  # generic license recorded as B


def test_add_components_on_three_versions():
    cands, jobs = _candidates(), _jobs()
    pairs = pl.DataFrame(
        {
            "CANDIDATE_ID": [1, 2, 3, 3],
            "Job_id": ["ORD_1", "ORD_1", "ORD_1", "ORD_2"],
            "distance_km": [5.0, 15.0, 22.0, 40.0],
        }
    )
    jr = job_requirements(jobs)
    assert jr["job_category"].to_list() == ["cassiere", "muratore"]
    p = add_components(pairs, build_profile_real(cands), jr, suffix="_P")
    assert p["Card_match_P"].to_list() == [1.0, 0.0, 1.0, 1.0]
    assert p["Languages_match_P"].to_list() == [1.0, 0.0, 0.0, 1.0]
    assert p["match_skills_P"].to_list() == pytest.approx([2 / 3, 0.0, 1 / 3, 0.0])
    assert p["Distance_match_P"].to_list() == [1.0, 0.6, 0.2, 0.0]
    o = add_components(pairs, _profile_original(), jr, suffix="_O")
    assert o["Card_match_O"].to_list() == [1.0, 0.0, 0.0, 0.0]  # cand 3 licenses missed
    assert o["match_skills_O"].to_list() == p["match_skills_P"].to_list()  # unchanged


def test_calibration_table_groups_by_raw_similarity():
    pairs = pl.DataFrame(
        {
            "Experience_raw_P": [0.0, 0.5, 0.65, 0.85, 1.0, 1.0],
            "Experience_hat_P": [0.0, 0.9, 0.9, 0.95, 1.0, 1.0],
            "Experience": [0.0, 0.9, 0.9, 0.95, 1.0, 1.0],
        }
    )
    out = calibration_table(pairs)
    assert out["group"].to_list()[0] == "no titles (raw = 0)"
    assert out["group"].to_list()[-1] == "verbatim match (raw = 1)"
    assert out["n_pairs"].sum() == 6
    verbatim = out.filter(pl.col("group") == "verbatim match (raw = 1)")
    assert verbatim["n_pairs"][0] == 2 and verbatim["calibrated_mean"][0] == 1.0
