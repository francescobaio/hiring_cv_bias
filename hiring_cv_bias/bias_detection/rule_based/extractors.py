import re

from hiring_cv_bias.bias_detection.rule_based.patterns import (
    LANGUAGE_REGEXES_EN,
    LANGUAGE_VARIANTS,
    driver_license_pattern_eng,
)


def extract_driver_license(text: str) -> set[str]:
    return (
        {"driver_license"} if driver_license_pattern_eng.search(text.lower()) else set()
    )


_driver_code = re.compile(r"^[a-e][1-9]?e?$", re.IGNORECASE)  # "B", "B1", "CE"


def norm_driver_license(skill: str | None) -> str:
    if not isinstance(skill, str):
        return ""
    skill = skill.lower().strip()

    if _driver_code.match(skill) or driver_license_pattern_eng.search(skill):
        return "driver_license"
    return skill


def extract_languages(text: str) -> set[str]:
    text = text.lower()
    return {lang for lang, rx in LANGUAGE_REGEXES_EN.items() if rx.search(text)}


_reverse_language_map: dict[str, str] = {
    variant.lower(): code
    for code, variants in LANGUAGE_VARIANTS.items()
    for variant in variants
}

_missing: set[str] = set()


def norm_languages(skill: str | None) -> str:
    if not isinstance(skill, str):
        return ""

    raw = skill.strip()
    key = raw.lower()
    code = _reverse_language_map.get(key)

    if not code:
        _missing.add(raw)

    return code or ""
