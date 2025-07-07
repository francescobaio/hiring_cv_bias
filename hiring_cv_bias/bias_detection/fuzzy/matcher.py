from typing import Set

from sentence_transformers import SentenceTransformer, util


class SemanticMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def semantic_comparison(
        self,
        custom_skills: Set[str],
        parser_skills: Set[str],
        threshold: float = 0.7,
    ) -> Set[str]:
        if len(custom_skills) != 0 and len(parser_skills) != 0:
            custom_skills_list = list(custom_skills)
            parser_skills_list = list(parser_skills)

            custom_embeddings = self.model.encode(
                custom_skills_list,
                convert_to_tensor=True,
            )

            parser_embeddings = self.model.encode(
                parser_skills_list,
                convert_to_tensor=True,
            )

            similarities = util.cos_sim(custom_embeddings, parser_embeddings)
            custom_matches = set()
            parser_matches = set()

            for i in range(similarities.shape[0]):
                for j in range(similarities.shape[1]):
                    if similarities[i][j] >= threshold:
                        custom_matches.add(custom_skills_list[i])
                        parser_matches.add(parser_skills_list[j])
                        break

            custom_skills = custom_skills - custom_matches
            custom_skills.update(parser_matches)

        return custom_skills
