import re
import string
from fuzzywuzzy import process, fuzz
from hiring_cv_bias.config import JOB_TITLES


def clean_cv(cv: str) -> str:
    cv = cv.replace("\n", " ")
    cv = cv.replace("CV anonimizzato:", "")
    cv = cv.replace('"""', "")
    cv = re.sub(" +", " ", cv)
    cv = cv.replace(" ,", ",")
    cv = cv.lower()
    return cv


def clean_punctuation(text: str) -> str:
    to_remove = string.punctuation
    for punctuation in to_remove:
        if punctuation in text:
            text = text.replace(punctuation, '')
    return text

def parse_driving_license(text):
    matched_jobs = []
    text = clean_punctuation(text).split()
    for i in range(len(text)):
        first_word = text[i].strip()
        default_match = process.extractOne(first_word, JOB_TITLES) 
        ratio_match = process.extractOne(first_word, JOB_TITLES, scorer=fuzz.ratio)
        if default_match[0] == ratio_match[0] and ratio_match[1] >= 60 and default_match[1] >= 80:
            matched_jobs.append((first_word, ratio_match[0]))

    return list(set(matched_jobs))



