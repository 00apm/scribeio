"""Transcript cleanup: drop hallucinations, hesitations and repeated loops.

Settings come from the ``transcribe.cleanup`` config section. Steps, in order:

1. energy  - segment whose words are mostly unvoiced (silent track under them)
2. phrase  - short segment containing a known Whisper hallucination phrase
3. filler  - segment made only of hesitations (yyy, mmm, uh, um, hmm)
4. loop    - the same text repeated by one speaker more times than allowed

Only whole segments are dropped, and every drop is returned with its reason
so the report can list it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from scribe_rpg.transcribe.session import Segment

# Hesitation tokens (after normalize): hm/hmm; English uh, um, uhm, er, erm
# (also drawn out); Polish y/e/m and their mixes - checked in _filler_token.
_HESITATION = re.compile(r"h+m+|u+h+m*|u+m+|e+r+m*")
_YEM = re.compile(r"[yem]+")
_DOUBLED = re.compile(r"(.)\1")


@dataclass
class Dropped:
    """A segment removed from the transcript.

    Attributes:
        segment: The removed segment.
        reason: ``"energy"``, ``"phrase"``, ``"filler"`` or ``"loop"``.
        detail: Extra information for the report, e.g. ``"25% voiced"``.
    """

    segment: Segment
    reason: str
    detail: str = ""


def normalize(text: str) -> str:
    """Lower-case `text`, turn punctuation into spaces, collapse whitespace."""
    return " ".join(re.sub(r"[^\w\s]|_", " ", text.lower()).split())


def is_filler(normalized: str) -> bool:
    """Tell whether normalized text is nothing but hesitations.

    Answers such as "mhm", "aha", "tak", "no" and short words such as "my"
    or "u" are not hesitations.
    """
    tokens = normalized.split()
    return bool(tokens) and all(_filler_token(t) for t in tokens)


def _filler_token(token: str) -> bool:
    if _HESITATION.fullmatch(token):
        return True
    # Polish hesitations are y/e/m runs, but "my" is a real word: require a
    # single letter ("y") or a doubled letter ("yyy", "eem", "mmm").
    if not _YEM.fullmatch(token):
        return False
    return len(token) == 1 or _DOUBLED.search(token) is not None


def voiced_ratio(segment: Segment) -> float:
    """Share of the segment's words that are voiced (0.0 without words)."""
    if not segment.words:
        return 0.0
    return sum(w.voiced for w in segment.words) / len(segment.words)


def apply_cleanup(
    segments: list[Segment], cfg: dict[str, Any]
) -> tuple[list[Segment], list[Dropped]]:
    """Run all cleanup steps.

    Args:
        segments: Raw segments of all tracks.
        cfg: The ``transcribe.cleanup`` settings.

    Returns:
        The kept segments (ordered by start) and the dropped ones with
        their reasons.
    """
    phrases = [normalize(p) for p in cfg["phrases"]]
    kept: list[Segment] = []
    dropped: list[Dropped] = []
    for s in segments:
        reason = _check(s, cfg, phrases)
        if reason:
            dropped.append(Dropped(s, *reason))
        else:
            kept.append(s)
    kept, loops = _drop_loops(kept, cfg["loop_min_words"], cfg["loop_max_repeat"])
    return kept, dropped + loops


def _check(
    s: Segment, cfg: dict[str, Any], phrases: list[str]
) -> tuple[str, str] | None:
    """Return (reason, detail) if the segment must go, else None."""
    ratio = voiced_ratio(s)
    if ratio < cfg["voiced_ratio"]:
        return "energy", f"{ratio:.0%} voiced"
    text = normalize(s.text)
    short = len(text.split()) <= cfg["phrase_max_words"]
    if short and any(p and p in text for p in phrases):
        return "phrase", ""
    if not cfg["keep_fillers"] and is_filler(text):
        return "filler", ""
    return None


def _drop_loops(
    segments: list[Segment], min_words: int, max_repeat: int
) -> tuple[list[Segment], list[Dropped]]:
    """Drop repeats beyond `max_repeat` of the same text by the same speaker.

    Repeats are counted per speaker over that speaker's consecutive
    segments, so other speakers talking in between do not break a loop.
    Short texts (fewer than `min_words`) are never treated as loops.
    """
    kept: list[Segment] = []
    dropped: list[Dropped] = []
    last: dict[str, tuple[str, int]] = {}  # account -> (text, run length)
    for s in sorted(segments, key=lambda s: (s.start, s.account)):
        text = normalize(s.text)
        prev, run = last.get(s.account, ("", 0))
        run = run + 1 if text == prev else 1
        last[s.account] = (text, run)
        if run > max_repeat and len(text.split()) >= min_words:
            dropped.append(Dropped(s, "loop", f"repeat {run}"))
        else:
            kept.append(s)
    return kept, dropped
