"""Estimate of the ``Experience`` component: the highest cosine similarity between the job
category and the candidate's job titles under a multilingual sentence encoder
(``intfloat/multilingual-e5-base``). The encoder was chosen on the P profile, where Adecco's
value is known, against three other encoders and a string-matching baseline (section 1.3 of
the notebook). Its cosine lives on the same scale as Adecco's value, so the estimate is the
cosine itself, with no calibration step."""

import difflib
import os
from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np
import polars as pl
from lingua import Language, LanguageDetectorBuilder
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 256

# Encoders compared on P before choosing ``MODEL_NAME`` (display name -> Hugging Face id).
CANDIDATE_MODELS = {
    "MiniLM-L12": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet-base": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "LaBSE": "sentence-transformers/LaBSE",
    "e5-base": MODEL_NAME,
    "e5-large": "intfloat/multilingual-e5-large",
    "bge-m3": "BAAI/bge-m3",
    # Italian-only encoders
    "it: sentence-BERTino": "efederici/sentence-BERTino",
    "it: sbert-italian-xxl": "nickprock/sentence-bert-base-italian-xxl-uncased",
}

# (title, job category) pairs that a good estimator must rate as close.
PROBE_PAIRS = [
    ("autista", "autista di ambulanza"),
    ("commesso", "commesso di punto vendita/shop assistant"),
    ("commesso", "commesso di supermercato"),
    ("cameriere", "cameriere di sala"),
    ("ingegnere", "ingegnere meccanico"),
    ("rappresentante di vendita", "sales representative"),
    ("warehouse worker", "magazziniere"),
]


def _pair_texts(
    pairs: pl.DataFrame, profile: pl.DataFrame, jobs_req: pl.DataFrame
) -> tuple[list[list[str]], list[str]]:
    """The candidate's titles and the job category of every pair, in the order of ``pairs``."""
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
    titles = [t or [] for t in df["experience"].to_list()]
    categories = [c or "" for c in df["job_category"].to_list()]
    return titles, categories


class TitleSimilarity:
    """The estimation rule shared by every estimator: 1 if the job category is among the
    candidate's titles, 0 if the candidate has no titles, otherwise the highest similarity
    between the category and the titles. Subclasses define the similarity of one pair of texts."""

    def prepare(self, texts: Iterable[str]) -> None:
        """Hook run once on every text before the pair similarities are computed."""

    def n_parameters(self) -> int | None:
        """Size of the underlying model, ``None`` when there is none."""
        return None

    def pair_similarity(self, title: str, category: str) -> float:
        raise NotImplementedError

    def similarity(
        self, pairs: pl.DataFrame, profile: pl.DataFrame, jobs_req: pl.DataFrame
    ) -> np.ndarray:
        titles, categories = _pair_texts(pairs, profile, jobs_req)
        self.prepare(categories)
        self.prepare(t for ts in titles for t in ts)
        out = np.zeros(len(titles), dtype=np.float32)
        for i, (ts, cat) in enumerate(zip(titles, categories, strict=True)):
            if not ts or not cat:
                continue
            if cat in ts:
                out[i] = 1.0
                continue
            out[i] = max(self.pair_similarity(t, cat) for t in ts)
        return np.clip(out, 0.0, 1.0)


class LexicalSimilarity(TitleSimilarity):
    """String-matching baseline: the larger of the character-level ratio (``difflib``) and
    the Jaccard overlap of the words."""

    def pair_similarity(self, title: str, category: str) -> float:
        words_t = set(title.replace("/", " ").split())
        words_c = set(category.replace("/", " ").split())
        jaccard = (
            len(words_t & words_c) / len(words_t | words_c) if words_t | words_c else 0
        )
        ratio = difflib.SequenceMatcher(None, title, category).ratio()
        return max(ratio, jaccard)


class ExperienceEstimator(TitleSimilarity):
    """Cosine similarity of normalized sentence embeddings. Texts are encoded as they are
    (without the ``query:`` prefix suggested for e5, which scored slightly worse on P) and
    cached in ``cache_path`` so that repeated builds do not re-encode them."""

    def __init__(
        self, model_name: str = MODEL_NAME, cache_path: str | None = None
    ) -> None:
        self.model_name = model_name
        self.cache_path = cache_path
        self._embeddings: dict[str, np.ndarray] = {}
        self._model: Any = None
        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=False)
            self._embeddings = dict(
                zip(data["texts"].tolist(), data["vectors"], strict=True)
            )

    def _load(self) -> Any:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def n_parameters(self) -> int:
        return sum(int(p.numel()) for p in self._load().parameters())

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._load().encode(
            list(texts),
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def prepare(self, texts: Iterable[str]) -> None:
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

    def pair_similarity(self, title: str, category: str) -> float:
        self.prepare([title, category])
        return float(self._embeddings[title] @ self._embeddings[category])


def embedding_cache_path(cache_dir: str, model_name: str) -> str:
    return os.path.join(cache_dir, model_name.split("/")[-1] + ".npz")


def candidate_estimators(cache_dir: str) -> dict[str, TitleSimilarity]:
    """The string baseline and the encoders of ``CANDIDATE_MODELS``, with one cache each."""
    estimators: dict[str, TitleSimilarity] = {"lexical baseline": LexicalSimilarity()}
    for name, model in CANDIDATE_MODELS.items():
        estimators[name] = ExperienceEstimator(
            model, embedding_cache_path(cache_dir, model)
        )
    return estimators


def evaluate_estimate(pairs: pl.DataFrame, estimate: np.ndarray) -> dict[str, float]:
    """How close ``estimate`` is to the real ``Experience`` of ``pairs``: correlations on all
    pairs and on the pairs with titles but no verbatim match (``spearman_non_verbatim``),
    mean absolute error, the lowest estimate among the pairs with titles and the number of
    pairs estimated below 0.4 while the real value is above 0.8 (``n_far_below``)."""
    y = pairs["Experience"].to_numpy()
    with_titles = estimate > 0
    non_verbatim = with_titles & (estimate < 1)
    return {
        "pearson": cast(float, pearsonr(estimate, y)[0]),
        "spearman": cast(float, spearmanr(estimate, y)[0]),
        "spearman_non_verbatim": cast(
            float, spearmanr(estimate[non_verbatim], y[non_verbatim])[0]
        ),
        "mae": float(np.mean(np.abs(estimate - y))),
        "min_with_titles": float(estimate[with_titles].min()),
        "n_far_below": int(np.sum((estimate < 0.4) & (y > 0.8))),
    }


def compare_estimators(
    pairs: pl.DataFrame,
    profile: pl.DataFrame,
    jobs_req: pl.DataFrame,
    estimators: dict[str, TitleSimilarity],
) -> pl.DataFrame:
    """One row per estimator: its size in millions of parameters and the metrics of
    ``evaluate_estimate`` on ``pairs``."""
    rows = []
    for name, est in estimators.items():
        n = est.n_parameters()
        rows.append(
            {"estimator": name, "parameters_M": None if n is None else round(n / 1e6)}
            | evaluate_estimate(pairs, est.similarity(pairs, profile, jobs_req))
        )
    return pl.DataFrame(rows)


def probe_table(
    estimators: dict[str, TitleSimilarity],
    probes: Sequence[tuple[str, str]] = PROBE_PAIRS,
) -> pl.DataFrame:
    """The similarity each estimator gives to the (title, job category) ``probes``."""
    return pl.DataFrame(
        {
            "title": [t for t, _ in probes],
            "job_category": [c for _, c in probes],
        }
        | {
            name: [est.pair_similarity(t, c) for t, c in probes]
            for name, est in estimators.items()
        }
    )


def language_shares(texts: Sequence[str]) -> pl.DataFrame:
    """Share of ``texts`` that ``lingua`` reads as Italian or English, deciding between the
    two languages only. Mixed texts such as ``commesso di punto vendita/shop assistant``
    count as Italian."""
    detector = LanguageDetectorBuilder.from_languages(
        Language.ITALIAN, Language.ENGLISH
    ).build()
    detected = {t: detector.detect_language_of(t) for t in set(texts)}
    return pl.DataFrame(
        {
            "language": ["italian", "english"],
            "share": [
                sum(detected[t] == lang for t in texts) / len(texts)
                for lang in (Language.ITALIAN, Language.ENGLISH)
            ],
        }
    )
