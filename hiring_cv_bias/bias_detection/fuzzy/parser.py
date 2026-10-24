import nltk
import spacy
from nltk.corpus import stopwords
from spacy.matcher import PhraseMatcher


class JobParser:
    def __init__(
        self,
        job_list: list[str],
    ):
        nltk.download("stopwords", quiet=True)
        self.stopwords = stopwords.words("english")
        self.job_list = job_list
        self.spacy_model = spacy.load("en_core_web_sm")
        self.phrase_matcher = PhraseMatcher(self.spacy_model.vocab, attr="LOWER")
        patterns = [self.spacy_model.make_doc(job) for job in self.job_list]
        self.phrase_matcher.add("Jobs", patterns)

    def parse_with_n_grams(self, text: str) -> set[str]:
        jobs_found = set()

        doc = self.spacy_model(text)

        match = self.phrase_matcher(doc)

        if len(match) > 0:
            for _, start, end in match:
                jobs_found.add(doc[start:end].text)

        return jobs_found
