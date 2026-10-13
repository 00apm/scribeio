"""Merge pipeline: raw session + settings + glossary -> lines and report data.

Every decision is made here (cleanup, speaker names, lines, glossary
corrections, bleed), from data stored in ``<name>.json``. That is why
`scribe merge` can redo the whole transcript in seconds, without audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scribe_rpg.craig import sanitize, speaker_name
from scribe_rpg.transcribe.cleanup import Dropped, apply_cleanup
from scribe_rpg.transcribe.glossary import Glossary
from scribe_rpg.transcribe.lines import Bleed, Line, build_lines, find_bleed
from scribe_rpg.transcribe.session import RawSession, Segment


@dataclass
class TrackStats:
    """Report summary for one track.

    Attributes:
        account: Track account.
        speaker: Name shown in the transcript.
        kind: ``"player"`` or ``"music"``.
        kept: Segments kept.
        dropped: Segments dropped by cleanup.
        words: Words in kept segments.
        speaking_time: Sum of the track's line durations, in seconds.
    """

    account: str
    speaker: str
    kind: str
    kept: int = 0
    dropped: int = 0
    words: int = 0
    speaking_time: float = 0.0


@dataclass
class MergeResult:
    """Everything the transcript and the report need.

    Attributes:
        lines: Transcript lines, in order.
        speakers: Speaker names in order of their first line.
        dropped: Segments removed by cleanup, with reasons.
        bleed: Possible bleed between tracks.
        stats: One entry per track, in track order.
        warnings: Messages for the report (unknown accounts, cut prompt,
            estimated timestamps).
    """

    lines: list[Line]
    speakers: list[str]
    dropped: list[Dropped]
    bleed: list[Bleed]
    stats: list[TrackStats]
    warnings: list[str]

    def speaker_of(self, account: str) -> str:
        """Speaker name for a track account (the account if unknown)."""
        for s in self.stats:
            if s.account == account:
                return s.speaker
        return account


def run(
    raw: RawSession, cfg: dict[str, Any], speakers: dict[str, str], glossary: Glossary
) -> MergeResult:
    """Build the transcript and report data from a raw session.

    Args:
        raw: Session data as stored in the JSON (not modified).
        cfg: The ``transcribe`` settings section.
        speakers: Sanitized account -> speaker name.
        glossary: Corrections to apply to line text.

    Returns:
        Lines, drops, bleed, per-track stats and warnings.
    """
    b = cfg["bleed"]
    kept, dropped = apply_cleanup(raw.segments, cfg["cleanup"])
    lines = build_lines(kept, speakers, cfg["lines"]["pause"])
    for line in lines:
        line.text = glossary.correct(line.text)
    bleed = find_bleed(kept, b["similarity"], b["min_words"], b["min_overlap"])
    return MergeResult(
        lines=lines,
        speakers=list(dict.fromkeys(line.speaker for line in lines)),
        dropped=dropped,
        bleed=bleed,
        stats=_stats(raw, speakers, kept, dropped, lines),
        warnings=_warnings(raw, speakers),
    )


def _stats(
    raw: RawSession,
    speakers: dict[str, str],
    kept: list[Segment],
    dropped: list[Dropped],
    lines: list[Line],
) -> list[TrackStats]:
    """Per-track counts, in the recording's track order."""
    stats = {
        t.account: TrackStats(t.account, speaker_name(speakers, t.account), t.kind)
        for t in raw.tracks
    }

    def of(account: str) -> TrackStats:
        # A segment from an account missing in the track list still counts.
        default = TrackStats(account, speaker_name(speakers, account), "player")
        return stats.setdefault(account, default)

    for s in kept:
        of(s.account).kept += 1
        of(s.account).words += len(s.words)
    for d in dropped:
        of(d.segment.account).dropped += 1
    for line in lines:
        of(line.account).speaking_time += line.end - line.start
    return list(stats.values())


def _warnings(raw: RawSession, speakers: dict[str, str]) -> list[str]:
    """Warnings for the report, from the session data and speaker names."""
    out = []
    unknown = [
        t.account
        for t in raw.tracks
        if t.kind == "player" and sanitize(t.account) not in speakers
    ]
    if unknown:
        out.append(
            "unknown accounts (no speakers: entry, account name used): "
            + ", ".join(unknown)
        )
    cut = raw.whisper.get("prompt_cut") or []
    if cut:
        out.append(
            "glossary names left out of the Whisper prompt (prompt_max_tokens): "
            + ", ".join(cut)
        )
    estimated = sum(w.estimated for s in raw.segments for w in s.words)
    if estimated:
        out.append(
            f"{estimated} words had no timestamp from alignment; times estimated"
        )
    return out
