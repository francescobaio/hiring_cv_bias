from dataclasses import dataclass
from typing import Dict, List, NamedTuple


@dataclass(frozen=True)
class Conf:
    tp: int
    fp: int
    tn: int
    fn: int

    def __str__(self) -> str:
        return (
            "\n"
            f"Counts   -> TP:{self.tp:<5}  FP:{self.fp:<5} "
            f"TN:{self.tn:<5}  FN:{self.fn:<5}\n"
            f"Metrics  -> Precision:{self.precision:.2f}  "
            f"Recall:{self.equality_of_opportunity:.3f}  "
            f"F1:{self.f1:.3f}  "
            f"Acc:{self.accuracy:.3f}"
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.equality_of_opportunity,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def equality_of_opportunity(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.equality_of_opportunity
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def accuracy(self) -> float:
        tot = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / tot if tot else 0.0

    # Negative predicted value
    @property
    def calibration_npv(self) -> float:
        return self.tn / (self.tn + self.fn) if (self.tn + self.fn) else 0.0

    @property
    def conditional_use_error_neg(self) -> float:
        return self.fn / (self.tn + self.fn) if (self.tn + self.fn) else 0.0


class Result(NamedTuple):
    conf: Conf
    tp_rows: List[Dict]
    fp_rows: List[Dict]
    fn_rows: List[Dict]
    tn_rows: List[Dict]
