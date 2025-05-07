import re

import pycountry

RED = "\033[31m"
RESET = "\033[0m"

DRIVER_LICENSE_SNIPPET_WORDS = [
    "autonomo",
    "patente",
    "auto",
    "veicolo",
    "munito",
    "licenza",
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
    r"\b(?:A|B|C|D|E|AM|A1|A2|B1|C1|D1|BE|C1E|D1E|CE|DE)\b\s*(?:driving|car|category)?\s*licen[cs]e\b|"
    r"\bdriving\s+licen[cs]e\b|"
    r"\bown\s+car\b"
    r")",
    re.IGNORECASE | re.UNICODE,
)

driver_license_pattern_it = re.compile(
    r"(?:"
    r"\bpatent[ei]\b(?:\s*(?:di\s*(?:guida|categoria))?\s*[:\-]?\s*\w+)?|"
    r"\bautomunit[oa](?:\/[oa])?\b|"
    r"\b(?:patente(?:\s+di\s+guida)?|categoria|cat\.?|driving\s+licen[cs]e)\s*[:\-]?\s*[A-E][1-9]?\b"
    r"\bmunit[oa]\b"
    r")",
    re.IGNORECASE | re.UNICODE,
)

import re

from langcodes import Language

# Costruisce dizionario lingua -> varianti [inglese, italiano]
language_variants = {}

for lang in pycountry.languages:
    if hasattr(lang, "alpha_2"):
        lang_code = lang.alpha_2.lower()
        name_en = lang.name.lower()
        try:
            name_it = Language.get(lang_code).display_name("it").lower()
        except LookupError:
            name_it = None

        variants = {name_en, lang_code}
        if name_it:
            variants.add(name_it)

        language_variants[name_en] = sorted(variants)

# Costruzione regex
LANGUAGE_REGEXES = {
    lang: re.compile(
        rf"""
        \b(?:{"|".join(map(re.escape, variants))})\b
        |madrelingua\s+(?:{"|".join(map(re.escape, variants))})
        |lingua\s+(?:{"|".join(map(re.escape, variants))})
        |conoscenza\s+(?:della\s+)?(?:lingua\s+)?(?:{"|".join(map(re.escape, variants))})
        |certificazione\s+(?:di\s+)?(?:{"|".join(map(re.escape, variants))})
        |(?:{"|".join(map(re.escape, variants))})\s+(?:parlata|scritta|orale)
        |(?:{"|".join(map(re.escape, variants))})\s*(?:A1|A2|B1|B2|C1|C2|madrelingua)
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    for lang, variants in language_variants.items()
}
