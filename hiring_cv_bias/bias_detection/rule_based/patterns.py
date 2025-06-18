import re

import pycountry
from langcodes import Language

RED = "\033[31m"
RESET = "\033[0m"

DRIVER_LICENSE_SNIPPET_WORDS = [
    "self-employed",
    "license",
    "car",
    "vehicle",
    "equipped",
    "licence",
]
DRIVER_LICENSE_SNIPPET_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(word) for word in DRIVER_LICENSE_SNIPPET_WORDS)
    + r")\b",
    re.IGNORECASE,
)

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
