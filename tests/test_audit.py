import polars as pl

from hiring_cv_bias.reverse_matching.audit import (
    list_lengths,
    profile_overlap,
    version_differences,
)


def test_list_lengths():
    pairs = pl.DataFrame({"Job_id": ["a"] * 5 + ["b"] * 2 + ["c"] * 10})
    ll = list_lengths(pairs).to_dicts()[0]
    assert ll["jobs"] == 3 and ll["jobs_ge_5"] == 2 and ll["median"] == 5


def test_version_differences():
    df = pl.DataFrame(
        {
            "Card_match_O": [0.0, 1.0],
            "Card_match_C": [1.0, 1.0],
            "Languages_match_O": [1.0, 1.0],
            "Languages_match_C": [1.0, 1.0],
            "Regulated_Trainings_match_O": [1.0, 1.0],
            "Regulated_Trainings_match_C": [1.0, 1.0],
            "match_skills_O": [0.0, 0.0],
            "match_skills_C": [0.0, 0.0],
            "Distance_match_O": [1.0, 1.0],
            "Distance_match_C": [1.0, 1.0],
            "Experience_hat_O": [0.9, 0.9],
            "Experience_hat_C": [0.95, 0.9],
            "score_O": [3.0, 3.0],
            "score_C": [3.4, 3.0],
        }
    )
    vd = version_differences(df, "O", "C")
    assert vd.filter(pl.col("column") == "Card_match")["share_different"][0] == 0.5
    assert vd.filter(pl.col("column") == "score")["share_different"][0] == 0.5


def test_profile_overlap_counts_candidates_with_items_in_both():
    real = pl.DataFrame(
        {
            "CANDIDATE_ID": [1, 2, 3],
            "licenses": [["b"], ["a", "b"], []],
            "languages": [["inglese"], ["italiano"], ["francese"]],
            "experience": [["cassiere"], [], ["muratore"]],
        }
    )
    original = pl.DataFrame(
        {
            "CANDIDATE_ID": [1, 2, 3],
            "licenses": [["b"], ["b"], ["b"]],
            "languages": [["italiano"], ["italiano"], []],
            "experience": [["commesso", "cassiere"], ["cuoco"], ["muratore"]],
        }
    )
    out = profile_overlap(real, original).to_dicts()
    lic, lang, exp = out
    # the parser found a license for all three, candidate 3 has none in P
    assert lic["candidates_parser"] == 3 and lic["share_also_in_real"] == 2 / 3
    assert lic["share_common_item"] == 1.0 and lic["share_identical"] == 0.5
    # candidate 3 has no parser language; 1 and 2 are compared, only 2 agrees
    assert lang["candidates_parser"] == 2 and lang["share_also_in_real"] == 1.0
    assert lang["share_common_item"] == 0.5 and lang["share_identical"] == 0.5
    assert exp["candidates_parser"] == 3 and exp["share_also_in_real"] == 2 / 3
    assert exp["share_common_item"] == 1.0 and exp["share_identical"] == 0.5
