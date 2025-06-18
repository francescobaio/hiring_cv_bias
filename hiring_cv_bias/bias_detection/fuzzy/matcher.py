from typing import Dict, List, Union

import polars as pl
from sentence_transformers import SentenceTransformer, util
from tqdm.notebook import tqdm

from hiring_cv_bias.bias_detection.fuzzy.utils import remove_duplicates


class SemanticMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def semantic_comparison(
        self,
        custom_extracted_data: pl.DataFrame,
        parser_extracted_data: pl.DataFrame,
        threshold: float = 0.7,
    ) -> pl.DataFrame:
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
        matching_data: Dict[str, List[Union[str, bool, None]]] = {
            "CANDIDATE_ID": [],
            "Custom_Job": [],
            "Parser_Job": [],
            "Match": [],
        }
        for id in tqdm(
            common_ids, desc="Job Matching...", leave=True, total=len(common_ids)
        ):
            custom_extracted_data_filtered = custom_extracted_data.filter(
                custom_extracted_data[custom_extracted_data.columns[0]] == id
            )[custom_extracted_data.columns[1]].to_list()
            parser_extracted_data_filtered = parser_extracted_data.filter(
                parser_extracted_data[parser_extracted_data.columns[0]] == id
            )[parser_extracted_data.columns[1]].to_list()

            custom_extracted_data_processed = list(
                map(
                    lambda x: x.lower().strip(),
                    custom_extracted_data_filtered,
                )
            )

            parser_extracted_data_processed = list(
                map(
                    lambda x: x.lower().replace("(m/f)", "").strip(),
                    parser_extracted_data_filtered,
                )
            )

            custom_embeddings = self.model.encode(
                custom_extracted_data_processed,
                convert_to_tensor=True,
            )
            parser_embeddings = self.model.encode(
                parser_extracted_data_processed,
                convert_to_tensor=True,
            )
            similarities = util.cos_sim(custom_embeddings, parser_embeddings)
            custom_matches = []
            parser_matches = []
            for i in range(similarities.shape[0]):
                for j in range(similarities.shape[1]):
                    if similarities[i][j] >= threshold:
                        custom_matches.append(custom_extracted_data_processed[i])
                        parser_matches.append(parser_extracted_data_processed[j])
                        break

            custom_matches = remove_duplicates(custom_matches)
            parser_matches = remove_duplicates(parser_matches)
            num_custom_jobs = len(set(custom_extracted_data_processed))
            num_parser_jobs = len(set(parser_extracted_data_processed))
            max_num_jobs = max(num_custom_jobs, num_parser_jobs)
            matching_data["CANDIDATE_ID"] += [id] * max_num_jobs
            unmatched_custom_jobs = list(
                set(custom_extracted_data_processed) - set(custom_matches)
            )
            unmatched_parser_jobs = list(
                set(parser_extracted_data_processed) - set(parser_matches)
            )
            matching_data["Custom_Job"] += custom_matches + unmatched_custom_jobs
            matching_data["Parser_Job"] += parser_matches + unmatched_parser_jobs
            matching_data["Match"] += [True] * len(custom_matches) + [False] * (
                max_num_jobs - len(custom_matches)
            )
            if num_custom_jobs < max_num_jobs:
                matching_data["Custom_Job"] += [None] * (max_num_jobs - num_custom_jobs)
            else:
                matching_data["Parser_Job"] += [None] * (max_num_jobs - num_parser_jobs)

        return pl.DataFrame(matching_data)
