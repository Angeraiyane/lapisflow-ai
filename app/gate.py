"""Layer-1 lexical safety guard (deterministic). Patterns are FIXED before measuring.

Do NOT tune these patterns on the validation leaks: that would be fitting the guard to the
set used to judge it. If you change them after looking at results, say so in your notes.
"""
import re

LEXICAL_PATTERNS = [
    r"\b(lost|stolen|stole|missing)\b",
    r"\b(can't|cannot|can not|cant) find\b",
    r"\blocate\b",
    r"\bunauthori[sz]ed\b",
    r"\bfraud\w*\b",
    r"\bcompromised\b",
    r"\bnot recogni[sz]ed\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in LEXICAL_PATTERNS]


def check_lexical_gate(text: str) -> dict:
    """Return {'triggered': bool, 'pattern': str | None}."""
    for pattern in _COMPILED:
        if pattern.search(text):
            return {"triggered": True, "pattern": pattern.pattern}
    return {"triggered": False, "pattern": None}