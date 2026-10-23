"""Build ``data/ethics/pairs.parquet``: one row per (candidate, job) pair of the clean
candidates, without the score outliers, with the components, the estimated ``Experience``
and the re-computed score under the three profile versions P, O and C.
Run with ``python -m hiring_cv_bias.reverse_matching.build``."""

import os
from dataclasses import dataclass

import polars as pl

from hiring_cv_bias.config import (
    CLEANED_REVERSE_MATCHING_PATH,
    DATA_DIR,
    REVERSE_MATCHING_PATH,
)
from hiring_cv_bias.reverse_matching.components import add_components, job_requirements
from hiring_cv_bias.reverse_matching.experience import (
    MODEL_NAME,
    ExperienceEstimator,
    embedding_cache_path,
    evaluate_estimate,
)
from hiring_cv_bias.reverse_matching.loader import (
    dedupe_pairs,
    load_false_negatives,
    load_parser_output,
)
from hiring_cv_bias.reverse_matching.profiles import (
    build_profile_corrected,
    build_profile_original,
    build_profile_real,
)
from hiring_cv_bias.reverse_matching.scoring import (
    RULE_COMPONENTS,
    fit_weights,
    score_expr,
    score_outliers,
)
from hiring_cv_bias.utils import load_data, load_excel_sheets

ETHICS_DATA_DIR = DATA_DIR + "ethics/"
PAIRS_PATH = ETHICS_DATA_DIR + "pairs.parquet"
EMBEDDINGS_DIR = ETHICS_DATA_DIR + "embeddings/"


@dataclass
class BuildResult:
    pairs: pl.DataFrame
    profiles: dict[str, pl.DataFrame]
    experience_diagnostics: dict[str, float]


def build_dataset() -> BuildResult:
    raw = load_excel_sheets(
        REVERSE_MATCHING_PATH, ["ReverseMatching", "Candidates", "Jobs"]
    )
    pairs = dedupe_pairs(raw["ReverseMatching"])
    pairs = pairs.filter(~score_outliers(pairs))
    # the recovered formula: least-squares weights rounded to one decimal
    weights = {k: round(v, 1) for k, v in fit_weights(pairs).items()}
    clean = load_data(CLEANED_REVERSE_MATCHING_PATH)
    pairs = pairs.join(
        clean.select("CANDIDATE_ID", "Gender", "Location"), on="CANDIDATE_ID"
    )

    jobs_req = job_requirements(raw["Jobs"])
    profiles = {"P": build_profile_real(raw["Candidates"])}
    profiles["O"] = build_profile_original(profiles["P"], load_parser_output())
    profiles["C"] = build_profile_corrected(
        profiles["O"],
        load_false_negatives("driving_license"),
        load_false_negatives("language_skill"),
        load_false_negatives("job_title"),
    )
    for v, prof in profiles.items():
        pairs = add_components(pairs, prof, jobs_req, suffix=f"_{v}")

    # the same estimator for every version; its accuracy is measured on P
    estimator = ExperienceEstimator(
        MODEL_NAME, embedding_cache_path(EMBEDDINGS_DIR, MODEL_NAME)
    )
    hat = {
        v: estimator.similarity(pairs, prof, jobs_req) for v, prof in profiles.items()
    }
    diagnostics = evaluate_estimate(pairs, hat["P"])
    pairs = pairs.with_columns(
        pl.Series(f"Experience_hat_{v}", hat[v]) for v in profiles
    )
    # the formula on each version's components; Interest is the same in every version
    pairs = pairs.with_columns(
        score_expr(
            weights,
            {c: f"{c}_{v}" for c in RULE_COMPONENTS}
            | {"Experience": f"Experience_hat_{v}"},
        ).alias(f"score_{v}")
        for v in profiles
    )

    os.makedirs(ETHICS_DATA_DIR, exist_ok=True)
    pairs.write_parquet(PAIRS_PATH)
    for v, prof in profiles.items():
        prof.write_parquet(ETHICS_DATA_DIR + f"profile_{v}.parquet")
    return BuildResult(pairs, profiles, diagnostics)


if __name__ == "__main__":
    result = build_dataset()
    print(f"written {PAIRS_PATH}: {result.pairs.height} pairs")
    print(f"Experience estimator on P: {result.experience_diagnostics}")
