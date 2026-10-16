"""Write ``<name>_transcript.md``: YAML frontmatter plus one line per utterance.

The transcript has no headings; structuring it (chapters, summaries) is left
to the LLM assistant that reads it::

    ---
    date: 2026-01-01
    start: "19:00"
    duration: "3:50:50"
    speakers: ["Game Master", "Player One"]
    source: "session.zip"
    ---

    [01:02:10] Game Master: Wchodzicie do piwnicy, jest ciemno i
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from scribe_rpg.transcribe.pipeline import MergeResult
from scribe_rpg.transcribe.session import RawSession


def hms(seconds: float) -> str:
    """Format a time as ``hh:mm:ss`` (seconds rounded down)."""
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def duration(seconds: float) -> str:
    """Format a length as ``h:mm:ss`` (rounded to the nearest second)."""
    s = round(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _q(value: str) -> str:
    """Quote a value for YAML.

    A JSON string is a valid double-quoted YAML scalar, so names containing
    ``:``, ``#``, ``[`` or quotes cannot break the frontmatter.
    """
    return json.dumps(value, ensure_ascii=False)


def render(raw: RawSession, result: MergeResult) -> str:
    """Return the transcript as Markdown text."""
    start = datetime.fromisoformat(raw.start)
    speakers = ", ".join(_q(s) for s in result.speakers)
    head = [
        "---",
        f"date: {raw.date}",
        f"start: {_q(start.strftime('%H:%M'))}",
        f"duration: {_q(duration(raw.duration))}",
        f"speakers: [{speakers}]",
        f"source: {_q(raw.source)}",
        "---",
        "",
    ]
    body = [f"[{hms(line.start)}] {line.speaker}: {line.text}" for line in result.lines]
    return "\n".join(head + body) + "\n"


def write(path: Path, raw: RawSession, result: MergeResult) -> None:
    """Write the transcript as UTF-8 with LF line endings."""
    Path(path).write_text(render(raw, result), encoding="utf-8", newline="\n")
