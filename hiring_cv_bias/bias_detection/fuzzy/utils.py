from typing import List


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
