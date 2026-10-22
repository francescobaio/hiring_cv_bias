"""Estimate of the ``Experience`` component: cosine similarity between the job category and
the candidate's job titles (multilingual sentence embeddings), calibrated to the real values
on the P profile with isotonic regression (section 1.3 of the notebook)."""

import os
from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.isotonic import IsotonicRegression

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 256


class ExperienceEstimator:
    def __init__(self, cache_path: str | None = None) -> None:
        self.cache_path = cache_path
        self._embeddings: dict[str, np.ndarray] = {}
        self._model: Any = None
        self.calibrator: IsotonicRegression | None = None
        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=False)
            self._embeddings = dict(
                zip(data["texts"].tolist(), data["vectors"], strict=True)
            )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            self._model = SentenceTransformer(MODEL_NAME)
        vectors = self._model.encode(
            list(texts),
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _embed(self, texts: Iterable[str]) -> None:
        """Embed and cache the texts not seen before."""
        new = sorted({t for t in texts if t and t not in self._embeddings})
        if not new:
            return
        vectors = self._encode(new)
        self._embeddings.update(zip(new, vectors, strict=True))
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            keys = list(self._embeddings)
            np.savez(
                self.cache_path,
                texts=np.array(keys),
                vectors=np.stack([self._embeddings[k] for k in keys]),
            )

    def raw_similarity(
        self, pairs: pl.DataFrame, profile: pl.DataFrame, jobs_req: pl.DataFrame
    ) -> np.ndarray:
        """1 if the job category is among the titles, 0 without titles, else the best cosine."""
        df = (
            pairs.select("CANDIDATE_ID", "Job_id")
            .join(
                profile.select("CANDIDATE_ID", "experience"),
                on="CANDIDATE_ID",
                how="left",
                maintain_order="left",
            )
            .join(
                jobs_req.select("Job_id", "job_category"),
                on="Job_id",
                how="left",
                maintain_order="left",
            )
        )
        titles: list[list[str]] = [t or [] for t in df["experience"].to_list()]
        categories: list[str] = [c or "" for c in df["job_category"].to_list()]
        self._embed(categories)
        self._embed(t for ts in titles for t in ts)

        out = np.zeros(len(titles), dtype=np.float32)
        for i, (ts, cat) in enumerate(zip(titles, categories, strict=True)):
            if not ts or not cat:
                continue
            if cat in ts:
                out[i] = 1.0
                continue
            m = np.stack([self._embeddings[t] for t in ts])
            out[i] = float(np.max(m @ self._embeddings[cat]))
        return np.clip(out, 0.0, 1.0)

    def calibrate(self, raw: np.ndarray, target: np.ndarray) -> None:
        """Fit the monotone map raw similarity -> real ``Experience``."""
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.calibrator.fit(raw, target)

    def predict(self, raw: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("call calibrate() first")
        return np.asarray(self.calibrator.predict(raw), dtype=np.float64)


def calibration_table(pairs: pl.DataFrame) -> pl.DataFrame:
    """On P: raw similarity, real ``Experience`` and calibrated estimate by band of raw similarity."""
    raw = pl.col("Experience_raw_P")
    group = (
        pl.when(raw == 0)
        .then(pl.lit("no titles (raw = 0)"))
        .when(raw == 1)
        .then(pl.lit("verbatim match (raw = 1)"))
        .otherwise(
            raw.cut(
                [0.6, 0.7, 0.8, 0.9],
                labels=[
                    "(0, 0.6)",
                    "[0.6, 0.7)",
                    "[0.7, 0.8)",
                    "[0.8, 0.9)",
                    "[0.9, 1)",
                ],
                left_closed=True,
            ).cast(pl.String)
        )
    )
    return (
        pairs.group_by(group.alias("group"))
        .agg(
            pl.len().alias("n_pairs"),
            raw.mean().alias("raw_mean"),
            pl.col("Experience").mean().alias("real_mean"),
            pl.col("Experience_hat_P").mean().alias("calibrated_mean"),
        )
        .sort("raw_mean")
    )


def evaluate_estimate(pairs: pl.DataFrame, estimate: np.ndarray) -> dict[str, float]:
    """Correlations, mean absolute error and share of real ``Experience = 1`` pairs
    estimated at 0.99 or above."""
    y = pairs["Experience"].to_numpy()
    return {
        "pearson": cast(float, pearsonr(estimate, y)[0]),
        "spearman": cast(float, spearmanr(estimate, y)[0]),
        "mae": float(np.mean(np.abs(estimate - y))),
        "share_exact_1_recovered": float(np.mean(estimate[y >= 0.9999] >= 0.99)),
    }
