"""Layer-1 lexical safety guard (deterministic, runs BEFORE the probability gate).

HISTORY (29 Oct 2026)
- v1 patterns (first 7) were fixed BEFORE measuring: caught 44/156 protected, 21/1806 false alarms.
- The last 4 patterns were added AFTER seeing the validation leaks (Khan: harden the guard).
  => validation numbers for this guard are no longer an unbiased estimate. The untouched TEST set
  is the only honest check. Say so in the README and the survey.
"""
import re

GUARD_VERSION = "lex_v2_11patterns"  # bump when the pattern list changes (stored in every audit row)

LEXICAL_PATTERNS = [
    r"\b(lost|stolen|stole|missing)\b",
    r"\b(can't|cannot|can not|cant) find\b",
    r"\blocate\b",
    r"\bunauthori[sz]ed\b",
    r"\bfraud\w*\b",
    r"\bcompromised\b",
    r"\bnot recogni[sz]ed\b",
    # added after seeing validation leaks (Khan's list):
    r"\blocation\b",
    r"\bmisplaced\b",
    r"\bwhere (is|are) my card\b",
    r"\b(can't|cannot|can not|cant) see my card\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in LEXICAL_PATTERNS]


def check_lexical_gate(text: str) -> dict:
    """Return {'triggered': bool, 'pattern': str | None}."""
    for pattern in _COMPILED:
        if pattern.search(text):
            return {"triggered": True, "pattern": pattern.pattern}
    return {"triggered": False, "pattern": None}