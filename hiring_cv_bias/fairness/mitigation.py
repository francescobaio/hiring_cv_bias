"""A toy example of the mitigation techniques, on a matcher that can be trained (phase 4 of
the notebooks).

Adecco's matcher is a formula with known weights, so the mitigations of the course have
nothing to train, and phase 3 shows that flipping its decisions can only level the North
down. To show the techniques where they apply, ``toy_candidates`` draws the candidates of
one opening, with a known truth (``hired_truth``), past decisions biased against the
South (``hired_past``) and a proxy of the region (``distance`` from the opening, as
``Distance_match`` in Adecco's formula), and a logistic regression (``fit_logistic``,
``decide``) learns the matcher from the past decisions without reading the region: the
bias comes in through the proxy, as fairness through unawareness fails to prevent. One mitigation per level of the taxonomy:
``reweighing_weights`` reweighs the training rows (pre-processing), ``fit_logistic`` with
a ``penalty`` adds the demographic parity term of the Prejudice Remover to the loss
(in-processing), ``equalized_odds_flips`` and ``apply_flips`` flip the decisions at random
per group (post-processing, Hardt et al. 2016; phase 3 applies the same linear program
to Adecco's decisions, where it can only level the North down). ``mitigation_table``
evaluates every decision against the truth with the metrics of phase 3."""

from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy.optimize import linprog, minimize

from hiring_cv_bias.fairness.metrics import (
    REFERENCE,
    parser_effect,
    per_group,
    selection_rates,
    share,
    with_weights,
)
from hiring_cv_bias.fairness.protocol import SEED

# the columns of the design matrix after the intercept: the two scores a recruiter reads
# and the distance from the opening; the region itself is not a feature
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


def design_matrix(df: pl.DataFrame) -> np.ndarray:
    """Build the input matrix of the logistic regression from the candidates table: one
    row per candidate, with a column of ones for the intercept followed by the columns
    ``TOY_FEATURES`` (``skills``, ``experience``, ``distance``). The region is not among
    them. ``fit_logistic`` trains on this matrix and ``decide`` predicts from it."""
    return df.select(pl.lit(1.0), *TOY_FEATURES).to_numpy()


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    group: np.ndarray | None = None,
    penalty: float = 0.0,
) -> np.ndarray:
    """The coefficients of a logistic regression of ``y`` on ``x``, by minimizing the
    weighted cross-entropy (``weights`` are 1 by default) plus ``penalty`` times the
    demographic parity term of the Prejudice Remover: the squared difference between the
    mean predicted probability of the rows in ``group`` and of the others. With
    ``penalty`` 0 it is the plain regression. The gradient is analytic and the
    minimizer is L-BFGS."""
    w = np.ones(len(y)) if weights is None else weights
    in_group = np.zeros(len(y), dtype=bool) if group is None else group
    w_sum = w.sum()

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        p = sigmoid(x @ beta)
        p_safe = np.clip(p, 1e-12, 1 - 1e-12)
        loss = -np.sum(w * (y * np.log(p_safe) + (1 - y) * np.log(1 - p_safe))) / w_sum
        grad = x.T @ (w * (p - y)) / w_sum
        if penalty:
            diff = p[in_group].mean() - p[~in_group].mean()
            d_diff = np.where(in_group, 1 / in_group.sum(), -1 / (~in_group).sum())
            loss += penalty * diff**2
            grad = grad + penalty * (x.T @ (2 * diff * d_diff * p * (1 - p)))
        return loss, grad

    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    return result.x


def coefficients_table(
    coefficients: dict[str, np.ndarray], digits: int = 2
) -> pl.DataFrame:
    """The fitted coefficients side by side, one column per model, one row per term."""
    return pl.DataFrame(
        {"term": ["intercept", *TOY_FEATURES]}
        | {name: beta.round(digits) for name, beta in coefficients.items()}
    )


def decide(
    df: pl.DataFrame, beta: np.ndarray, name: str, threshold: float = THRESHOLD
) -> pl.DataFrame:
    """Add the decision ``hired_<name>``: the model with coefficients ``beta`` hires a
    candidate when its predicted probability passes ``threshold``."""
    hired = sigmoid(design_matrix(df) @ beta) > threshold
    return df.with_columns(pl.Series(f"hired_{name}", hired))


def reweighing_weights(
    df: pl.DataFrame, attribute: str = "Location", label: str = "hired_past"
) -> pl.DataFrame:
    """The weights of Reweighing (Kamiran & Calders 2012): one row per (group, label) cell
    with P(A = a), P(Y = y), P(A = a, Y = y) and w = P(A = a) P(Y = y) / P(A = a, Y = y).
    A cell is weighed above 1 when it is rarer than independence would make it, so that
    in the weighted data the attribute and the label are independent."""
    n = pl.len()
    return (
        df.select(attribute, label)
        .with_columns(
            (pl.len().over(attribute) / n).alias("p_group"),
            (pl.len().over(label) / n).alias("p_label"),
            (pl.len().over(attribute, label) / n).alias("p_joint"),
        )
        .unique()
        .with_columns(
            (pl.col("p_group") * pl.col("p_label") / pl.col("p_joint")).alias("weight")
        )
        .sort(attribute, label, descending=[False, True])
    )


def row_weights(
    df: pl.DataFrame, weights: pl.DataFrame, attribute: str = "Location"
) -> np.ndarray:
    """The weight of every row of ``df``, from the table of ``reweighing_weights``."""
    label = weights.columns[1]
    return (
        df.join(
            weights.select(attribute, label, "weight"),
            on=[attribute, label],
            maintain_order="left",
        )
        .get_column("weight")
        .to_numpy()
    )


def equalized_odds_flips(
    df: pl.DataFrame,
    decision: str,
    label: str = "truth",
    outcome: str = "hired",
    attribute: str = "Location",
) -> pl.DataFrame:
    """The post-processing of Hardt et al. 2016, on the decision ``<outcome>_<decision>``
    against ``<outcome>_<label>``. A selected candidate of group a stays selected with
    probability ``keep``, a rejected one becomes selected with probability ``add``; the
    rates after the flips are keep · TPR + add · (1 − TPR) and keep · FPR + add ·
    (1 − FPR). The probabilities are chosen by a linear program: equal TPR and FPR in
    every group, with the fewest expected errors against the label. The table gives, per
    group, the rates before and after and the two probabilities."""
    d, y = pl.col(f"{outcome}_{decision}"), pl.col(f"{outcome}_{label}")
    rates = per_group(
        df,
        [attribute],
        [
            pl.col("w").sum().alias("_w"),
            share(y).alias("_p_label"),
            share(d, among=y).alias("tpr"),
            share(d, among=~y).alias("fpr"),
        ],
    )
    # one variable per (group, decision): the probability of being selected after the
    # flip, in the order add, keep of each group
    tpr, fpr = rates["tpr"].to_numpy(), rates["fpr"].to_numpy()
    after_tpr = np.column_stack([1 - tpr, tpr])  # coefficients of (add, keep)
    after_fpr = np.column_stack([1 - fpr, fpr])
    k = len(rates)
    p_group = (rates["_w"] / rates["_w"].sum()).to_numpy()[:, None]
    p_label = rates["_p_label"].to_numpy()[:, None]
    cost = (p_group * (-p_label * after_tpr + (1 - p_label) * after_fpr)).ravel()
    equal = np.zeros((2 * (k - 1), 2 * k))
    for g in range(1, k):
        equal[2 * (g - 1), :2] = after_tpr[0]
        equal[2 * (g - 1), 2 * g : 2 * g + 2] = -after_tpr[g]
        equal[2 * g - 1, :2] = after_fpr[0]
        equal[2 * g - 1, 2 * g : 2 * g + 2] = -after_fpr[g]
    solution = linprog(cost, A_eq=equal, b_eq=np.zeros(2 * (k - 1)), bounds=(0, 1))
    add, keep = solution.x.reshape(k, 2).T
    return rates.with_columns(
        pl.Series("keep", keep),
        pl.Series("add", add),
        pl.Series("tpr_after", keep * tpr + add * (1 - tpr)),
        pl.Series("fpr_after", keep * fpr + add * (1 - fpr)),
    ).drop("attribute", "_w", "_p_label")


def apply_flips(
    df: pl.DataFrame,
    flips: pl.DataFrame,
    decision: str,
    name: str,
    outcome: str = "hired",
    attribute: str = "Location",
    seed: int = SEED,
) -> pl.DataFrame:
    """Add the decision ``<outcome>_<name>``: ``<outcome>_<decision>`` flipped at random
    with the probabilities of ``equalized_odds_flips``, one draw per candidate."""
    probability = (
        df.join(
            flips.select(pl.col("group").alias(attribute), "keep", "add"),
            on=attribute,
            how="left",
            maintain_order="left",
        )
        .select(pl.when(pl.col(f"{outcome}_{decision}")).then("keep").otherwise("add"))
        .to_series()
        .to_numpy()
    )
    draw = np.random.default_rng(seed).random(df.height)
    return df.with_columns(pl.Series(f"{outcome}_{name}", draw < probability))


def mitigation_table(
    df: pl.DataFrame,
    methods: Sequence[str],
    label: str = "truth",
    outcome: str = "hired",
    attribute: str = "Location",
    group: str = "SOUTH",
) -> pl.DataFrame:
    """Every decision ``<outcome>_<method>`` against ``<outcome>_<label>``, one row per
    method: the accuracy, the selection rate of the reference group and of ``group`` with
    the SPD and DI of ``group`` (``selection_rates``), and their false negative and false
    positive rates with the gaps of ``group`` (``parser_effect``)."""
    rates = selection_rates(df, [attribute], methods, outcome).filter(
        pl.col("group") == group
    )
    rows = []
    for m in methods:
        effect = parser_effect(df, [attribute], outcome, decision=m, label=label)
        wide = effect.select(
            pl.lit(m).alias("method"),
            "group",
            pl.col(f"rate_{m}").alias("rate"),
            "fnr",
            "fpr",
        ).pivot(on="group", index="method", values=["rate", "fnr", "fpr"])
        gaps = effect.filter(pl.col("group") == group).select("fnr_diff", "fpr_diff")
        accuracy = with_weights(df).select(
            share(pl.col(f"{outcome}_{m}") == pl.col(f"{outcome}_{label}")).alias(
                "accuracy"
            )
        )
        rows.append(
            pl.concat(
                [
                    wide,
                    accuracy,
                    rates.select(
                        pl.col(f"spd_{m}").alias("spd"), pl.col(f"di_{m}").alias("di")
                    ),
                    gaps,
                ],
                how="horizontal",
            )
        )
    regions = [REFERENCE[attribute], group]
    return pl.concat(rows).select(
        "method",
        "accuracy",
        *[f"rate_{r}" for r in regions],
        "spd",
        "di",
        *[f"fnr_{r}" for r in regions],
        "fnr_diff",
        *[f"fpr_{r}" for r in regions],
        "fpr_diff",
    )
