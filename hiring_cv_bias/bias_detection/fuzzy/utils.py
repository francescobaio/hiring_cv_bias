from typing import Any, List, Optional

import numpy as np
import polars as pl


def normalize_jobs(jobs: List[str]) -> List[str]:
    for job in jobs.copy():
        words = job.split()
        if len(words) > 3:
            jobs.remove(job)
        elif "/" in job:
            first_job, second_job = job.split("/")
            jobs.remove(job)
            jobs.extend([first_job, second_job])
    return jobs


def compute_tp_rate(data: pl.DataFrame) -> Optional[float]:
    fn = data.filter(pl.col("False_Negative")).height
    tp = data.filter(pl.col("True_Positive")).height
    if tp + fn != 0:
        rate = np.round((tp / (tp + fn)) * 100, 1)
    else:
        rate = None
    return rate


def remove_duplicates(list: List[Any]) -> List[Any]:
    """
    Efficiently removes duplicate elements from a list while preserving the original order of items.
    """
    seen = set()
    return [x for x in list if x not in seen and not bool(seen.add(x))]  # type: ignore
