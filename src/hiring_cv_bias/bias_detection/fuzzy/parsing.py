import re
import string


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
            text = text.replace(punctuation, "")
    return text
