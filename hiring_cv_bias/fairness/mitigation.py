"""A toy example of the mitigation techniques, on a matcher that can be trained (phase 4 of
the notebooks), with the libraries of the course: scikit-learn for the matcher, fairlib for
the pre- and in-processing, fairlearn for the post-processing and the metrics.

Adecco's matcher is a formula with known weights, so the mitigations of the course have
nothing to train, and phase 3 shows that flipping its decisions can only level the North
down. To show the techniques where they apply, ``toy_candidates`` draws the candidates of
one opening, with a known truth (``hired_truth``), past decisions biased against the
South (``hired_past``) and a proxy of the region (``distance`` from the opening, as
``Distance_match`` in Adecco's formula). A logistic regression (``fit_matcher``,
``decide``) learns the matcher from the past decisions without reading the region: the
bias comes in through the proxy, as fairness through unawareness fails to prevent. One
mitigation per level of the taxonomy: ``reweighing_weights`` reweighs the training rows
with fairlib's Reweighing (pre-processing); ``fit_prejudice_remover`` trains the same
matcher with fairlib's Prejudice Remover, a mutual-information term in the loss
(in-processing); ``equalized_odds_on_decisions`` and ``equalized_odds_on_score`` fit
fairlearn's ThresholdOptimizer, which flips the decisions at random per group, as Hardt
et al. 2016 (post-processing; phase 3 applies the first to Adecco's decisions, where it
can only level the North down), with ``flip_table`` to read its probabilities and
``apply_equalized_odds`` to draw the decisions. ``mitigation_table`` evaluates every
decision against the truth with fairlearn's MetricFrame, in the metrics of phase 3."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import fairlib as fl
import numpy as np
import pandas as pd
import polars as pl
import torch
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlib.inprocessing.prejudice_remover import PrejudiceRemover
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch import nn

from hiring_cv_bias.fairness.metrics import GROUP_ORDER, REFERENCE
from hiring_cv_bias.fairness.protocol import SEED

# the features of the matcher: the two scores a recruiter reads and the distance from the
# opening; the region itself is not a feature
TOY_FEATURES = ("skills", "experience", "distance")
# the range of ``distance`` in each region: the opening is in the North, so northern
# candidates live closer on average, with an overlap between the two
DISTANCE_RANGE = {"NORTH": (0.0, 0.6), "SOUTH": (0.4, 1.0)}
THRESHOLD = 0.5


def toy_candidates(
    n: int,
    seed: int = SEED,
    south_share: float = 0.4,
    rejected_share: float = 0.3,
    noise: float = 0.15,
) -> pl.DataFrame:
    """Generate the candidates of one job opening, with random draws from ``seed``.
    Returns a table with one row per candidate and the columns:

    - ``Location``: NORTH or SOUTH, SOUTH with probability ``south_share``;
    - ``skills``, ``experience``: two scores uniform in [0, 1], independent of the
      region;
    - ``distance``: how far the candidate lives from the opening, 0 next door and 1
      far away, uniform in ``DISTANCE_RANGE`` of the region. The opening is in the
      North, so the distance is a proxy of the region;
    - ``hired_truth``: True when ``skills + experience + noise > 1``, with Gaussian
      noise of standard deviation ``noise``. It depends neither on the region nor on
      the distance;
    - ``hired_past``: what the recruiters decided in the past. It equals the truth,
      except that a share ``rejected_share`` of the qualified southern candidates, drawn
      at random, were rejected. This is the only bias of the toy."""
    rng = np.random.default_rng(seed)
    south = rng.random(n) < south_share
    skills, experience = rng.random(n), rng.random(n)
    lo, hi = (
        np.array([DISTANCE_RANGE["SOUTH" if s else "NORTH"][i] for s in south])
        for i in (0, 1)
    )
    distance = lo + (hi - lo) * rng.random(n)
    truth = skills + experience + rng.normal(0, noise, n) > 1
    wrongly_rejected = south & (rng.random(n) < rejected_share)
    return pl.DataFrame(
        {
            "Location": np.where(south, "SOUTH", "NORTH"),
            "skills": skills,
            "experience": experience,
            "distance": distance,
            "hired_truth": truth,
            "hired_past": truth & ~wrongly_rejected,
        }
    )


def features(df: pl.DataFrame) -> pd.DataFrame:
    """The input of the matcher: the columns ``TOY_FEATURES`` of the candidates table, as
    the pandas DataFrame scikit-learn, fairlib and fairlearn read. The region is not
    among them."""
    return df.select(TOY_FEATURES).to_pandas()


def labels(df: pl.DataFrame, label: str = "past", outcome: str = "hired") -> np.ndarray:
    """The column ``<outcome>_<label>`` as the 0/1 vector the libraries take as labels."""
    return df[f"{outcome}_{label}"].to_numpy().astype(int)


def fit_matcher(
    df: pl.DataFrame, weights: np.ndarray | None = None, label: str = "past"
) -> LogisticRegression:
    """A logistic regression of the decisions ``hired_<label>`` on the features, without
    regularization (``C`` infinite), with the row ``weights`` of reweighing if given."""
    return LogisticRegression(C=np.inf, max_iter=1000).fit(
        features(df), labels(df, label), sample_weight=weights
    )


@dataclass
class LinearMatcher:
    """A fitted logistic matcher in the form scikit-learn gives it: ``intercept_``,
    ``coef_`` and ``predict``. ``fit_prejudice_remover`` returns one, read off the torch
    model that fairlib trained, so that every matcher of the notebook is used alike."""

    intercept_: np.ndarray
    coef_: np.ndarray

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        z = np.asarray(x, dtype=float) @ self.coef_[0] + self.intercept_[0]
        return 1 / (1 + np.exp(-z)) > THRESHOLD


Matcher: TypeAlias = LogisticRegression | LinearMatcher


def coefficients_table(models: Mapping[str, Matcher], digits: int = 2) -> pl.DataFrame:
    """The fitted coefficients side by side, one column per model, one row per term."""
    return pl.DataFrame(
        {"term": ["intercept", *TOY_FEATURES]}
        | {
            name: np.concatenate([m.intercept_, m.coef_[0]]).round(digits)
            for name, m in models.items()
        }
    )


def decide(df: pl.DataFrame, model: Matcher, name: str) -> pl.DataFrame:
    """Add the decision ``hired_<name>`` of ``model`` for every candidate of ``df``."""
    hired = np.asarray(model.predict(features(df))).astype(bool)
    return df.with_columns(pl.Series(f"hired_{name}", hired))


def _fairlib_frame(
    df: pl.DataFrame, attribute: str, group: str, label: str, outcome: str
) -> fl.DataFrame:
    """The features with the sensitive attribute (1 for ``group``, 0 otherwise: fairlib
    takes it binary) and the label, as fairlib's DataFrame with ``sensitive`` and
    ``targets`` set."""
    frame = features(df)
    frame[attribute] = (df[attribute] == group).to_numpy().astype(float)
    frame[label] = labels(df, label, outcome).astype(float)
    out = fl.DataFrame(frame)
    # fairlib adds these properties to pandas' DataFrame at import time
    out.sensitive = {attribute}  # type: ignore[attr-defined]
    out.targets = {label}  # type: ignore[attr-defined]
    return out


def reweighing_weights(
    df: pl.DataFrame,
    attribute: str = "Location",
    group: str = "SOUTH",
    label: str = "past",
    outcome: str = "hired",
) -> np.ndarray:
    """The weight of every row of ``df`` by Reweighing (Kamiran & Calders 2012), as
    fairlib computes it: w(a, y) = P(A = a) P(Y = y) / P(A = a, Y = y) for the row's
    group ``a`` and label ``y``, above 1 for a cell rarer than independence would make
    it, so that in the weighted data the attribute and the label are independent."""
    weighted = fl.Reweighing().fit_transform(
        _fairlib_frame(df, attribute, group, label, outcome), favorable_label=1
    )
    return weighted["weights"].to_numpy()


def weights_table(
    df: pl.DataFrame,
    weights: np.ndarray,
    attribute: str = "Location",
    label: str = "past",
    outcome: str = "hired",
) -> pl.DataFrame:
    """The weights of ``reweighing_weights`` by cell: one row per (group, label) with
    the number of candidates and their common weight."""
    return (
        df.with_columns(pl.Series("weight", weights))
        .group_by(attribute, f"{outcome}_{label}")
        .agg(pl.len().alias("candidates"), pl.col("weight").first())
        .sort(attribute, f"{outcome}_{label}", descending=[False, True])
    )


class _Features(nn.Module):
    """A logistic regression on the first ``n`` columns of its input. fairlib passes the
    sensitive column with the features; the model leaves it unread, so that the region
    enters the loss and not the matcher."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x[:, : self.linear.in_features])).squeeze(-1)


def fit_prejudice_remover(
    df: pl.DataFrame,
    eta: float,
    attribute: str = "Location",
    group: str = "SOUTH",
    label: str = "past",
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 0.05,
    seed: int = SEED,
) -> LinearMatcher:
    """The matcher trained by fairlib's Prejudice Remover (Kamishima et al. 2012): the
    same logistic regression as ``fit_matcher``, fitted by Adam over ``epochs`` passes in
    batches of ``batch_size``, on the cross-entropy of ``hired_<label>`` plus ``eta``
    times the mutual information between the predicted probability and the sensitive
    attribute (1 for ``group``), computed on every batch. With ``eta`` 0 it is the plain
    regression. The label is passed as a vector: fairlib's loss needs it in one
    dimension, and the target of its DataFrame comes out as a column."""
    torch.manual_seed(seed)
    model = _Features(len(TOY_FEATURES))
    remover = PrejudiceRemover(
        model, optimizer=torch.optim.Adam(model.parameters(), lr=lr), eta=eta
    )
    remover.fit(
        _fairlib_frame(df, attribute, group, label, "hired"),
        y=labels(df, label).astype(float),
        epochs=epochs,
        batch_size=batch_size,
    )
    return LinearMatcher(
        model.linear.bias.detach().numpy().copy(),
        model.linear.weight.detach().numpy().copy(),
    )


class Decisions(BaseEstimator, ClassifierMixin):
    """An estimator whose prediction is its input, a column of 0/1 decisions. It lets
    ThresholdOptimizer post-process decisions already taken, as Hardt et al. 2016 do
    with a binary predictor, instead of thresholding a score."""

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "Decisions":
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x).ravel()


def decision_input(
    df: pl.DataFrame, decision: str, outcome: str = "hired"
) -> np.ndarray:
    """The decision ``<outcome>_<decision>`` as the one-column 0/1 matrix that
    ``Decisions`` passes through."""
    return labels(df, decision, outcome).reshape(-1, 1)


def _equalized_odds(
    estimator: BaseEstimator,
    predict_method: Literal["predict", "predict_proba"],
    prefit: bool,
) -> ThresholdOptimizer:
    return ThresholdOptimizer(
        estimator=estimator,
        constraints="equalized_odds",
        objective="accuracy_score",
        predict_method=predict_method,
        prefit=prefit,
    )


def equalized_odds_on_decisions(
    df: pl.DataFrame,
    decision: str,
    label: str = "truth",
    outcome: str = "hired",
    attribute: str = "Location",
) -> ThresholdOptimizer:
    """fairlearn's ThresholdOptimizer fitted on the decisions ``<outcome>_<decision>``
    of ``df`` against ``<outcome>_<label>``: per group of ``attribute``, the
    probabilities of keeping a selected pair and of selecting a rejected one that give
    every group the same true positive rate and the same false positive rate, with the
    fewest expected errors against the label (Hardt et al. 2016). ``flip_table`` reads
    the probabilities, ``apply_equalized_odds`` draws the decisions."""
    return _equalized_odds(Decisions(), "predict", prefit=False).fit(
        decision_input(df, decision, outcome),
        labels(df, label, outcome),
        sensitive_features=df[attribute].to_numpy(),
    )


def equalized_odds_on_score(
    df: pl.DataFrame,
    model: LogisticRegression,
    label: str = "truth",
    outcome: str = "hired",
    attribute: str = "Location",
) -> ThresholdOptimizer:
    """The same optimizer on the probability of ``model`` instead of its decisions: a
    threshold per group, mixed at random between two values, such that every group has
    the same rates against ``<outcome>_<label>``, with the fewest expected errors."""
    return _equalized_odds(model, "predict_proba", prefit=True).fit(
        features(df),
        labels(df, label, outcome),
        sensitive_features=df[attribute].to_numpy(),
    )


def apply_equalized_odds(
    df: pl.DataFrame,
    optimizer: ThresholdOptimizer,
    x: np.ndarray | pd.DataFrame,
    name: str,
    outcome: str = "hired",
    attribute: str = "Location",
    seed: int = SEED,
) -> pl.DataFrame:
    """Add the decision ``<outcome>_<name>``: the random draw of ``optimizer`` on ``x``,
    the decisions (``decision_input``) or the features (``features``) of ``df`` it was
    fitted for."""
    hired = optimizer.predict(
        x, sensitive_features=df[attribute].to_numpy(), random_state=seed
    )
    return df.with_columns(pl.Series(f"{outcome}_{name}", np.asarray(hired) == 1))


def _flip_probability(rule: Mapping, decision: float) -> float:
    """The probability that the rule of a group selects a pair with the given 0/1
    decision: with probability ``p_ignore`` the rule outputs a constant, otherwise one
    of two thresholds, drawn with probabilities ``p0`` and ``p1``."""
    thresholds = sum(
        float(rule[f"p{i}"]) * float(rule[f"operation{i}"](np.array([decision]))[0])
        for i in (0, 1)
    )
    p_ignore = float(rule.get("p_ignore", 0.0))
    return (
        p_ignore * float(rule.get("prediction_constant", 0.0))
        + (1 - p_ignore) * thresholds
    )


def flip_table(
    optimizer: ThresholdOptimizer,
    df: pl.DataFrame,
    decision: str,
    label: str = "truth",
    outcome: str = "hired",
    attribute: str = "Location",
) -> pl.DataFrame:
    """What an optimizer of ``equalized_odds_on_decisions`` does to each group of
    ``df``: the number of pairs and the true and false positive rates of the decision
    against the label (``tpr``, ``fpr``), the probability that a selected pair stays
    selected (``keep``) and that a rejected one becomes selected (``add``), and the
    rates after the flips, keep · TPR + add · (1 − TPR) and keep · FPR + add ·
    (1 − FPR), equal across the groups by construction."""
    rates = MetricFrame(
        metrics={
            "pairs": count,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
        },
        y_true=labels(df, label, outcome),
        y_pred=labels(df, decision, outcome),
        sensitive_features=df[attribute].to_numpy(),
    ).by_group
    rules = optimizer.interpolated_thresholder_.interpolation_dict
    rows = []
    for group, rule in rules.items():
        tpr, fpr = float(rates.loc[group, "tpr"]), float(rates.loc[group, "fpr"])
        keep, add = _flip_probability(rule, 1.0), _flip_probability(rule, 0.0)
        rows.append(
            {
                "group": group,
                "pairs": int(rates.loc[group, "pairs"]),
                "tpr": tpr,
                "fpr": fpr,
                "keep": keep,
                "add": add,
                "tpr_after": keep * tpr + add * (1 - tpr),
                "fpr_after": keep * fpr + add * (1 - fpr),
            }
        )
    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    return pl.DataFrame(sorted(rows, key=lambda r: order.get(r["group"], len(order))))


MITIGATION_RATES = {
    "rate": selection_rate,
    "fnr": false_negative_rate,
    "fpr": false_positive_rate,
}


def mitigation_table(
    df: pl.DataFrame,
    methods: Sequence[str],
    label: str = "truth",
    outcome: str = "hired",
    attribute: str = "Location",
    group: str = "SOUTH",
) -> pl.DataFrame:
    """Every decision ``<outcome>_<method>`` against ``<outcome>_<label>``, one row per
    method, with fairlearn's MetricFrame: the accuracy, the selection rate of the
    reference group and of ``group`` with the SPD and DI of ``group``, and their false
    negative and false positive rates with the gaps of ``group``, the metrics of
    phase 3."""
    reference = REFERENCE[attribute]
    y = labels(df, label, outcome)
    rows = []
    for m in methods:
        pred = labels(df, m, outcome)
        by_group = MetricFrame(
            metrics=MITIGATION_RATES,
            y_true=y,
            y_pred=pred,
            sensitive_features=df[attribute].to_numpy(),
        ).by_group
        rates = {
            f"{r}_{g}": float(by_group.loc[g, r])
            for r in MITIGATION_RATES
            for g in (reference, group)
        }
        rows.append(
            {
                "method": m,
                "accuracy": float(accuracy_score(y, pred)),
                f"rate_{reference}": rates[f"rate_{reference}"],
                f"rate_{group}": rates[f"rate_{group}"],
                "spd": rates[f"rate_{group}"] - rates[f"rate_{reference}"],
                "di": rates[f"rate_{group}"] / rates[f"rate_{reference}"],
                f"fnr_{reference}": rates[f"fnr_{reference}"],
                f"fnr_{group}": rates[f"fnr_{group}"],
                "fnr_diff": rates[f"fnr_{group}"] - rates[f"fnr_{reference}"],
                f"fpr_{reference}": rates[f"fpr_{reference}"],
                f"fpr_{group}": rates[f"fpr_{group}"],
                "fpr_diff": rates[f"fpr_{group}"] - rates[f"fpr_{reference}"],
            }
        )
    return pl.DataFrame(rows)
