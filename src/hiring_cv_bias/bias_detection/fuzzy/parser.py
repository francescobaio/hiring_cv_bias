from typing import List, Set

import spacy
from hiring_cv_bias.bias_detection.fuzzy.utils import clean_ner_entity
from sentence_transformers import SentenceTransformer, util


class JobParser:
    def __init__(
        self,
        job_list: List[str],
        parsing_model_name: str = "model-best",
        clustering_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        misleading_words: Set[str] = {
            "diploma",
            "diplomas",
            "skill",
            "skills",
        },
    ):
        self.job_list = job_list
        self.parsing_model = spacy.load(parsing_model_name)
        self.clustering_model = SentenceTransformer(clustering_model_name)
        self.job_embeddings = self.clustering_model.encode(
            job_list, convert_to_tensor=True
        )
        self.misleading_words = misleading_words

    def parse(self, text: str, min_similarity: float = 0.4) -> List[str]:
        jobs_found = []
        for doc in self.parsing_model.pipe([text], disable=["tagger", "parser"]):
            for ent in doc.ents:
                print(ent.label_)
                if ent.label_ == "EXPERIENCE":
                    cleaned_entity = clean_ner_entity(ent.text)
                    print(cleaned_entity)
                    entity_splits = cleaned_entity.split(",")
                    entity_jobs = []
                    for split in entity_splits:
                        if not any(word in split for word in self.misleading_words):
                            entity_jobs.append(
                                self.clustering_model.encode(
                                    split, convert_to_tensor=True
                                )
                            )
                    jobs_found += entity_jobs
                    # job_embedding = self.clustering_model.encode(
                    #     cleaned_entity, convert_to_tensor=True
                    # )
                    # jobs_found.append(job_embedding)

        normalized_jobs = []
        for job in jobs_found:
            distances = util.pytorch_cos_sim(job, self.job_embeddings)
            best_match_idx = distances.argmax()
            print(self.job_list[best_match_idx], distances[0][best_match_idx])
            if distances[0][best_match_idx] > min_similarity:
                best_match = self.job_list[best_match_idx]
                normalized_jobs.append(best_match)

        return list(set(normalized_jobs))
