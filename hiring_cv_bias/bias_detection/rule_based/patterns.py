import re

import polars as pl
import pycountry
from langcodes import Language

from hiring_cv_bias.bias_detection.fuzzy.utils import normalize_jobs
from hiring_cv_bias.config import JOBS_PATH

RED = "\033[31m"
RESET = "\033[0m"

driver_license_pattern_eng = re.compile(
    r"(?:"
    r"(?:driver['’]?s?|driving|car|category)\s*licen[cs]e\s*[:\-]?\s*(?:type\s*)?[a-z]{1,2}\b|"
    r"\blicen[cs]e\s*[:\-]?\s*(?:[A-E][1-9]?|AM|A1|A2|B1|C1|D1|BE|CE|DE)\b|"
    r"\b(?:A|B|C|D|E|AM|A1|A2|B1|C1|D1|BE|C1E|D1E|CE|DE)\b\s*(?:driving|car|category)?\s*licen[cs]e\b|"
    r"\bdriving\s+licen[cs]e\b|"
    r"\bown\s+car\b"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# driver_license_pattern_it = re.compile(
#     r"(?:"
#     r"\bpatent[ei]\b(?:\s*(?:di\s*(?:guida|categoria))?\s*[:\-]?\s*\w+)?|"
#     r"\bautomunit[oa](?:\/[oa])?\b|"
#     r"\b(?:patente(?:\s+di\s+guida)?|categoria|cat\.?|driving\s+licen[cs]e)\s*[:\-]?\s*[A-E][1-9]?\b"
#     r"\bmunit[oa]\b"
#     r")",
#     re.IGNORECASE | re.UNICODE,
# )

# ----------------------------------------------------------

# LANGUAGE_REGEXES = {
#     lang: re.compile(
#         rf"""
#         \b(?:{"|".join(map(re.escape, variants))})\b
#         |madrelingua\s+(?:{"|".join(map(re.escape, variants))})
#         |lingua\s+(?:{"|".join(map(re.escape, variants))})
#         |conoscenza\s+(?:della\s+)?(?:lingua\s+)?(?:{"|".join(map(re.escape, variants))})
#         |certificazione\s+(?:di\s+)?(?:{"|".join(map(re.escape, variants))})
#         |(?:{"|".join(map(re.escape, variants))})\s+(?:parlata|scritta|orale)
#         |(?:{"|".join(map(re.escape, variants))})\s*(?:A1|A2|B1|B2|C1|C2|madrelingua)
#         """,
#         re.IGNORECASE | re.VERBOSE,
#     )
#     for lang, variants in LANGUAGE_VARIANTS.items()
# }

LANGUAGE_VARIANTS = {}

for lang in pycountry.languages:
    if not hasattr(lang, "alpha_2"):
        continue

    # ISO code & name
    code = lang.alpha_2.lower()
    name_en = lang.name.lower()

    try:
        native = Language.get(code).display_name(code).lower()
    except LookupError:
        native = ""

    variants = {name_en}
    if native:
        variants.add(native)

    LANGUAGE_VARIANTS[code] = sorted(variants)

LANGUAGE_REGEXES_EN = {
    lang: re.compile(
        rf"""
        \b(?:{"|".join(map(re.escape, variants))})\b
        |native speaker(?: of)?\s+(?:{"|".join(map(re.escape, variants))})
        |languages?\s+(?:{"|".join(map(re.escape, variants))})
        |knowledge\s+of\s+(?:the\s+)?(?:language\s+)?(?:{"|".join(map(re.escape, variants))})
        |proficiency\s+in\s+(?:{"|".join(map(re.escape, variants))})
        |certification(?: in)?\s+(?:{"|".join(map(re.escape, variants))})
        |(?:{"|".join(map(re.escape, variants))})\s+(?:spoken|written|oral)
        |(?:{"|".join(map(re.escape, variants))})\s*(?:A1|A2|B1|B2|C1|C2|native)
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    for lang, variants in LANGUAGE_VARIANTS.items()
}


languages_pattern_eng = re.compile(
    "|".join(f"({pat.pattern})" for pat in LANGUAGE_REGEXES_EN.values()),
    re.IGNORECASE | re.VERBOSE,
)


jobs = pl.read_csv(JOBS_PATH)["preferredLabel"].to_list()
normalized_jobs = normalize_jobs(jobs)

jobs_pattern = re.compile("|".join(rf"\b{job}\b" for job in normalized_jobs))
