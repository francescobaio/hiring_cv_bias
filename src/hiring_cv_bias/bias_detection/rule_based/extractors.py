from __future__ import annotations

import re
from typing import Optional, Set

from .patterns import (
    LANGUAGE_REGEXES,
    driver_license_pattern_eng,
)


def extract_driver_license(text: str) -> Set[str]:
    return (
        {"driver_license"} if driver_license_pattern_eng.search(text.lower()) else set()
    )


def extract_languages(text: str) -> Set[str]:
    text = text.lower()
    return {lang for lang, rx in LANGUAGE_REGEXES.items() if rx.search(text)}


# ---------- normalization helpers ----------
_driver_code = re.compile(r"^[a-e][1-9]?e?$", re.I)  # e.g. "B", "B1", "CE"


def norm_driver_license(skill: Optional[str]) -> str:
    """
    Map any variant found in the parser output to 'driver_license'.

    Examples of matches:
      * B, B1, C, CE …
      * "driving licence", "driver's license"
    """
    if not isinstance(skill, str):
        return ""
    skill = skill.lower().strip()

    if _driver_code.match(skill) or driver_license_pattern_eng.search(skill):
        return "driver_license"
    return skill
