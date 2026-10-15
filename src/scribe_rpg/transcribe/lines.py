"""Build transcript lines from kept segments, and find possible bleed.

Lines are built from words, not segments, in three steps:

1. Per speaker, consecutive words form an utterance; a pause of at least
   ``transcribe.lines.pause`` seconds starts a new one.
2. An utterance is split where another speaker's utterance starts inside
   it, so a short "Tak." lands in the middle of a monologue instead of
   after it.
3. All parts are sorted by start time; each part becomes one line.

Bleed (the same speech picked up on two tracks) is only reported, never
removed.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

from scribe_rpg.craig import speaker_name
from scribe_rpg.transcribe.cleanup import normalize
from scribe_rpg.transcribe.session import Segment, Word

# Tolerance for the pause test: 2.3 - 0.8 is 1.4999999999999998 in floats.
_EPS = 1e-6


@dataclass
class Line:
    """One transcript line.

    Attributes:
        start: Start of the first word, seconds from the session start.
        end: End of the last word.
        account: Track account.
        speaker: Name shown in the transcript.
        text: The words joined with spaces (glossary corrections are applied
            later, by the pipeline).
    """

    start: float
    end: float
    account: str
    speaker: str
    text: str


@dataclass
class Bleed:
    """Two segments on different tracks with nearly the same text and time.

    Attributes:
        start: Start of the earlier segment.
        a: The segment that starts first.
        b: The other segment.
        similarity: Text similarity, 0..1.
    """

    start: float
    a: Segment
    b: Segment
    similarity: float


@dataclass
class _Part:
    """An utterance, or a piece of one after an interjection split."""

    account: str
    words: list[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


def build_lines(
    segments: list[Segment], speakers: dict[str, str], pause: float
) -> list[Line]:
    """Turn kept segments of all tracks into ordered transcript lines.

    Args:
        segments: Kept segments (any order, any tracks).
        speakers: Sanitized account -> speaker name; unknown accounts keep
            their name.
        pause: Seconds of silence that start a new line for a speaker.

    Returns:
        Lines sorted by start; ties by earlier end, then speaker name.
    """

    def name(account: str) -> str:
        return speaker_name(speakers, account)

    parts = _split_interjections(_utterances(segments, pause))
    parts.sort(key=lambda p: (p.start, p.end, name(p.account)))
    return [
        Line(
            p.start,
            p.end,
            p.account,
            name(p.account),
            " ".join(w.word for w in p.words),
        )
        for p in parts
    ]


def _utterances(segments: list[Segment], pause: float) -> list[_Part]:
    """Group each speaker's words into utterances separated by pauses."""
    by_account: dict[str, list[Word]] = defaultdict(list)
    for s in segments:
        by_account[s.account].extend(s.words)
    utts: list[_Part] = []
    for account, words in by_account.items():
        words.sort(key=lambda w: (w.start, w.end))
        current: list[Word] = []
        for w in words:
            if current and w.start - current[-1].end >= pause - _EPS:
                utts.append(_Part(account, current))
                current = []
            current.append(w)
        if current:
            utts.append(_Part(account, current))
    return utts


def _split_interjections(utts: list[_Part]) -> list[_Part]:
    """Split each utterance where another speaker's utterance begins inside it.

    The cut goes before the first word that starts at or after the other
    utterance's start. Pieces without words are dropped.
    """
    # Utterance starts sorted once; bisect finds the ones inside each
    # utterance without comparing every pair.
    starts = sorted((u.start, u.account) for u in utts)
    keys = [s for s, _ in starts]
    out: list[_Part] = []
    for u in utts:
        lo, hi = bisect_right(keys, u.start), bisect_left(keys, u.end)
        cuts = sorted({s for s, a in starts[lo:hi] if a != u.account})
        out.extend(_split(u, cuts))
    return out


def _split(u: _Part, cuts: list[float]) -> list[_Part]:
    """Cut `u` before the first word at or after each time in `cuts`."""
    parts: list[_Part] = []
    current: list[Word] = []
    i = 0
    for w in u.words:
        cut = False
        while i < len(cuts) and w.start >= cuts[i]:
            cut = True
            i += 1
        if cut and current:
            parts.append(_Part(u.account, current))
            current = []
        current.append(w)
    if current:
        parts.append(_Part(u.account, current))
    return parts


def find_bleed(
    segments: list[Segment], similarity: float, min_words: int, min_overlap: float
) -> list[Bleed]:
    """Find segments on two tracks that are the same speech.

    Args:
        segments: Kept segments of all tracks.
        similarity: Minimum text similarity (difflib ratio on normalized
            text), ``transcribe.bleed.similarity``.
        min_words: Both segments need at least this many words.
        min_overlap: Overlap in time, as a fraction of the shorter segment.

    Returns:
        Pairs ordered by start; ``a`` is the segment that starts first.
    """
    texts = {id(s): normalize(s.text) for s in segments}
    cands = sorted(
        (s for s in segments if len(texts[id(s)].split()) >= min_words),
        key=lambda s: (s.start, s.account),
    )
    found: list[Bleed] = []
    for i, a in enumerate(cands):
        for b in cands[i + 1 :]:
            # Sorted by start: once b starts after a ends, no later b overlaps.
            if b.start >= a.end:
                break
            if b.account == a.account:
                continue
            shorter = min(a.end - a.start, b.end - b.start)
            overlap = min(a.end, b.end) - b.start
            if shorter <= 0 or overlap < min_overlap * shorter:
                continue
            ratio = SequenceMatcher(None, texts[id(a)], texts[id(b)]).ratio()
            if ratio >= similarity:
                found.append(Bleed(a.start, a, b, round(ratio, 2)))
    return found
