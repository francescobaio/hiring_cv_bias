import re


def clean_cv(cv: str) -> str:
    cv = cv.replace("CV anonimizzato:", "")
    cv = cv.replace('"""', "")
    cv = re.sub(r"[\n\r]+", " ", cv)
    cv = re.sub(r"\s{2,}", " ", cv)
    return cv.strip()
