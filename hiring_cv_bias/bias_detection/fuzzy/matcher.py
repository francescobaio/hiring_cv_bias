import polars as pl
from sentence_transformers import SentenceTransformer, util


class SemanticMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def semantic_comparison(
        self,
        custom_extracted_data: pl.DataFrame,
        parser_extracted_data: pl.DataFrame,
        threshold: float = 0.7,
    ):
        custom_extracted_data_cleaned = custom_extracted_data.filter(
            pl.col(custom_extracted_data.columns[1]).is_not_null()
        )
        parser_extracted_data_cleaned = parser_extracted_data.filter(
            pl.col(parser_extracted_data.columns[1]).is_not_null()
        )
        custom_ids = set(
            custom_extracted_data_cleaned[
                custom_extracted_data_cleaned.columns[0]
            ].to_list()
        )
        parser_ids = set(
            parser_extracted_data_cleaned[
                parser_extracted_data_cleaned.columns[0]
            ].to_list()
        )
        common_ids = custom_ids & parser_ids
        for id in common_ids:
            custom_extracted_data_filtered = custom_extracted_data.filter(
                custom_extracted_data[custom_extracted_data.columns[0]] == id
            )[custom_extracted_data.columns[1]].to_list()
            parser_extracted_data_filtered = parser_extracted_data.filter(
                parser_extracted_data[parser_extracted_data.columns[0]] == id
            )[parser_extracted_data.columns[1]].to_list()

            parser_extracted_data_lower = list(
                map(lambda x: x.lower(), parser_extracted_data_filtered)
            )

            custom_embeddings = self.model.encode(
                custom_extracted_data_filtered,
                convert_to_tensor=True,
            )
            parser_embeddings = self.model.encode(
                parser_extracted_data_lower,
                convert_to_tensor=True,
            )
            similarities = util.cos_sim(custom_embeddings, parser_embeddings)
            custom_matches = set()
            parser_matches = set()
            for i in range(similarities.shape[0]):
                for j in range(similarities.shape[1]):
                    if similarities[i][j] >= threshold:
                        custom_matches.add(custom_extracted_data_filtered[i])
                        parser_matches.add(parser_extracted_data_lower[j])
                        print(
                            f"Custom job: {custom_extracted_data_filtered[i]}, Parser job: {parser_extracted_data_lower[j]}. Score: {similarities[i][j]}"
                        )
                        break
                    else:
                        print(
                            f"No match between custom job {custom_extracted_data_filtered[i]} and parser job {parser_extracted_data_lower[j]}. Score: {similarities[i][j]}"
                        )
