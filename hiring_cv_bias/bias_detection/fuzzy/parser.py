from typing import Any, Dict, List, Set

import nltk
import polars as pl
import spacy
from nltk.corpus import stopwords
from spacy.matcher import PhraseMatcher
from tqdm.notebook import tqdm


class JobParser:
    def __init__(
        self,
        job_list: List[str],
    ):
        nltk.download("stopwords")
        self.stopwords = stopwords.words("english")
        self.job_list = job_list
        self.spacy_model = spacy.load("en_core_web_sm")
        self.phrase_matcher = PhraseMatcher(self.spacy_model.vocab, attr="LOWER")
        patterns = [self.spacy_model.make_doc(job) for job in self.job_list]
        self.phrase_matcher.add("Jobs", patterns)

    def parse_with_n_grams(self, text: str) -> Set[str]:
        jobs_found = set()

        doc = self.spacy_model(text)

        match = self.phrase_matcher(doc)

        if len(match) > 0:
            for _, start, end in match:
                jobs_found.add(doc[start:end].text)

        return jobs_found

    def parse_df(self, df: pl.DataFrame) -> pl.DataFrame:
        jobs_data: Dict[str, List[Any]] = {"CANDIDATE_ID": [], "Job_Title": []}
        for row in tqdm(
            df.iter_rows(named=True), total=df.height, desc="Job parsing..."
        ):
            cv = row["Translated_CV"]
            jobs = self.parse_with_n_grams(cv)
            if len(jobs) > 0:
                jobs_data["Job_Title"].extend(jobs)
                jobs_data["CANDIDATE_ID"].extend([row["CANDIDATE_ID"]] * len(jobs))
            else:
                jobs_data["Job_Title"].append(None)
                jobs_data["CANDIDATE_ID"].append(row["CANDIDATE_ID"])

        return pl.DataFrame(jobs_data)
