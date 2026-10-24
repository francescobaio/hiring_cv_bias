import polars as pl

from hiring_cv_bias.fairness.protocol import (
    add_group,
    add_outcomes,
    filter_protocol,
    outcome_summary,
    protocol_sizes,
)


def _pairs() -> pl.DataFrame:
    # job a: 5 Female/Male + 1 Other (kept, 5 after the gender filter);
    # job b: 4 candidates (dropped); job c: 6 candidates
    return add_group(
        pl.DataFrame(
            {
                "Job_id": ["a"] * 6 + ["b"] * 4 + ["c"] * 6,
                "CANDIDATE_ID": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12],
                "Gender": ["Female", "Male", "Female", "Male", "Female", "Other"]
                + ["Male"] * 4
                + ["Female", "Male"] * 3,
                "Location": ["NORTH", "SOUTH", "NORTH", "CENTER", "SOUTH", "NORTH"]
                + ["NORTH"] * 4
                + ["SOUTH", "NORTH", "CENTER", "SOUTH", "NORTH", "CENTER"],
                # job a: candidates 2 and 3 tie on 4.0; job c: 9, 10 and 11 tie on 3.0
                # across the top-half cutoff (position 3)
                "score_adecco": [5.0, 4.0, 4.0, 3.0, 2.0, 9.0]
                + [1.0, 2.0, 3.0, 4.0]
                + [6.0, 5.0, 3.0, 3.0, 3.0, 1.0],
            }
        )
    )


def test_filter_protocol_drops_other_genders_and_short_lists():
    df = filter_protocol(_pairs())
    assert set(df["Job_id"]) == {"a", "c"}
    assert set(df["Gender"]) == {"Female", "Male"}
    assert df.filter(pl.col("Job_id") == "a")["n_in_job"].unique().to_list() == [5]


def test_outcomes_rank_within_the_job_and_break_ties_by_the_draw():
    df = add_outcomes(filter_protocol(_pairs()), ["score_adecco"])
    a = df.filter(pl.col("Job_id") == "a").sort("CANDIDATE_ID")
    # the two tied candidates get positions 2 and 3 in some order
    assert sorted(a["rank_adecco"].to_list()[1:3]) == [2, 3]
    assert a["rank_adecco"].to_list()[::3] == [1, 4]
    c = df.filter(pl.col("Job_id") == "c").sort("CANDIDATE_ID")
    # n = 6 → 3 selected: 7 and 8, and one of the three tied on 3.0
    assert c["top_half_adecco"].sum() == 3
    assert c["top_half_adecco"].to_list()[:2] == [True, True]
    assert c["top_half_adecco"].to_list()[2:5].count(True) == 1
    assert c["top3_adecco"].sum() == 3


def test_the_draw_is_shared_by_scores_and_independent_of_row_order():
    df = filter_protocol(_pairs()).with_columns(pl.col("score_adecco").alias("score_O"))
    out = add_outcomes(df, ["score_adecco", "score_O"])
    # identical scores give identical outcomes: the draw is the same for every score
    assert (out["top_half_adecco"] == out["top_half_O"]).all()
    shuffled = add_outcomes(df.reverse(), ["score_adecco", "score_O"])
    assert out["rank_adecco"].to_list() == shuffled["rank_adecco"].to_list()


def test_protocol_sizes_and_outcome_summary():
    df = filter_protocol(_pairs()).with_columns(
        (pl.col("score_adecco") + 1).alias("score_O")
    )
    df = add_outcomes(df, ["score_adecco", "score_O"])
    sizes = protocol_sizes(df, "Gender")
    assert sizes.filter(pl.col("Gender") == "all")["pairs"][0] == 11
    summary = outcome_summary(df, ["adecco", "O"])
    assert summary["agree_adecco"].to_list() == [1.0, 1.0]
    # only the three candidates of job c tied across the cutoff are decided by the draw
    assert summary["decided_by_draw"][0] == 3 / 11
