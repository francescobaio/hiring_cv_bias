from collections import namedtuple
from typing import Dict

Conf = namedtuple("Conf", "tp fp tn fn")


def scores(cm: Conf) -> Dict[str, float]:
    """Return accuracy / precision / recall / f1 as a dict."""
    total = sum(cm)
    precision = cm.tp / (cm.tp + cm.fp) if cm.tp + cm.fp else 0.0
    recall = cm.tp / (cm.tp + cm.fn) if cm.tp + cm.fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (cm.tp + cm.tn) / total if total else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
