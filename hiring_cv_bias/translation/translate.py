import os
import re

import polars as pl
from exploration.utils import load_data
from google.cloud import translate_v2 as translate
from src.hiring_cv_bias.config import CANDIDATE_CVS_PATH

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


def translate_text(text: str, target_language: str = "en") -> str:
    text = clean_cv_text(text)
    client = translate.Client()
    result = client.translate(text, target_language=target_language)
    return str(result["translatedText"])


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
