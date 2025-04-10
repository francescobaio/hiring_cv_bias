import re

RED = "\033[31m"
RESET = "\033[0m"

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
    r"(?i)\b(?:patente(?:\s+di\s+guida)?|categoria|cat\.?|driving\s+licen[cs]e)\s*[:\-]?\s*[A-E][1-9]?\b"
    r"\bmunit[oa]\b"
    r")",
    re.IGNORECASE | re.UNICODE,
)
