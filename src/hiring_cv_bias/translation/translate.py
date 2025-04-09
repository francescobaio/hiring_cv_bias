import os
import re

import polars as pl
from google.cloud import translate_v2 as translate
from hiring_cv_bias.config import CANDIDATE_CVS_PATH
from hiring_cv_bias.utils import load_data

import polars as pl
from google.cloud import translate_v2 as translate
from hiring_cv_bias.config import CANDIDATE_CVS_PATH
from hiring_cv_bias.utils import load_data

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "magnetic-market-455110-t2-0dc68481bf13.json"
)


def clean_cv_text(cv_text: str) -> str:
    cv_text = cv_text.replace("\n", " ")
    cv_text = cv_text.replace("CV anonimizzato:", "")
    cv_text = cv_text.replace('"""', "")
    cv_text = re.sub(" +", " ", cv_text)
    cv_text = cv_text.replace(" ,", ",")
    cv_text = cv_text.lower()
    return cv_text


def translate_text(text, target_language="en"):
    text = clean_cv_text(text)
    client = translate.Client()
    result = client.translate(text, target_language=target_language)
    return result["translatedText"]


if __name__ == "main":
    cv_data = load_data(CANDIDATE_CVS_PATH)
    cv_df_eng = cv_data.with_columns(
        pl.col("CV_text_anon")
        .map_elements(
            lambda x: translate_text(x, target_language="en"), return_dtype=pl.String
        )
        .alias("Translated_CV")
    )
    cv_df_eng.write_csv("Candidate_CVs_translated.csv", separator=";")
    print("Translations completed e saved in Candidate_CVs_translated.csv")
