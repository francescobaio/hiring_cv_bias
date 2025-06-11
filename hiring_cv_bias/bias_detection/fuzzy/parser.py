import math
from typing import Any, Dict, List

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

    def parse_with_n_grams(self, text: str) -> List[str]:
        jobs_found = []

        doc = self.spacy_model(text)

        match = self.phrase_matcher(doc)

        if len(match) > 0:
            for _, start, end in match:
                jobs_found.append(doc[start:end].text)

        return list(set(jobs_found))

    def _split_chunk(self, chunk: Any, max_len: int = 4) -> List[List[Any]]:
        chunk_pieces = [chunk]
        while len(chunk_pieces[0]) > max_len:
            new_chunk_pieces = []
            for piece in chunk_pieces:
                if len(piece.text) > 2:
                    if len(piece) > 3:
                        half_index = math.ceil(len(piece) / 2)
                        new_chunk_pieces.extend(
                            [piece[:half_index], piece[half_index:]]
                        )
                    else:
                        new_chunk_pieces.append(piece)
            chunk_pieces = new_chunk_pieces
        return chunk_pieces

    def _filter_short_words(self, chunk: Any) -> str:
        chunk_words = chunk.text.split()
        for word in chunk_words.copy():
            if len(word) < 2:
                chunk_words.remove(word)

        cleaned_chunk = " ".join(chunk_words)

        return cleaned_chunk

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
