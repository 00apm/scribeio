"""Write and read ``<name>_transcript.json``: the raw session data.

`scribe transcribe` writes the file once; `scribe merge` only reads it. The
file carries a format number so an incompatible future layout is refused
with a clear message instead of failing somewhere later.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scribeio.errors import ScribeError
from scribeio.transcribe.session import RawSession, Segment, TrackInfo, Word

FORMAT = 1


def write(path: Path, raw: RawSession) -> None:
    """Write `raw` as UTF-8 JSON (non-ASCII kept as is, times rounded to ms)."""
    data = {
        "format": FORMAT,
        "scribe": raw.scribe,
        "versions": raw.versions,
        "session": {
            "date": raw.date,
            "start": raw.start,
            "duration": _ms(raw.duration),
            "source": raw.source,
        },
        "whisper": raw.whisper,
        "warnings": raw.warnings,
        "tracks": [asdict(t) for t in raw.tracks],
        "segments": [_segment(s) for s in raw.segments],
    }
    text = json.dumps(data, ensure_ascii=False)
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def read(path: Path) -> RawSession:
    """Read a session file written by `write`.

    Raises:
        ScribeError: If the file is missing, is not JSON, has another
            format number, or misses required fields.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ScribeError(f"not found: {path.name}") from None
    except json.JSONDecodeError as e:
        raise ScribeError(f"{path.name}: not valid JSON (line {e.lineno})") from None
    fmt = data.get("format") if isinstance(data, dict) else None
    if fmt != FORMAT:
        raise ScribeError(
            f"{path.name}: unknown format {fmt!r} (this scribe reads format {FORMAT})"
        )
    try:
        s = data["session"]
        return RawSession(
            date=s["date"],
            start=s["start"],
            duration=s["duration"],
            source=s["source"],
            whisper=data["whisper"],
            tracks=[TrackInfo(**t) for t in data["tracks"]],
            segments=[
                Segment(**{**seg, "words": [Word(**w) for w in seg["words"]]})
                for seg in data["segments"]
            ],
            scribe=data.get("scribe", ""),
            versions=data.get("versions", {}),
            warnings=data.get("warnings", []),
        )
    # KeyError: a field is missing; TypeError: a field is unknown or the
    # value has the wrong shape (e.g. a list where an object was expected).
    except (KeyError, TypeError) as e:
        raise ScribeError(f"{path.name}: damaged session file ({e})") from None


def _ms(t: float) -> float:
    """Round a time in seconds to milliseconds."""
    return round(t, 3)


def _segment(s: Segment) -> dict[str, Any]:
    """Segment as a JSON object, keys in documented order, times in ms."""
    return {
        "account": s.account,
        "start": _ms(s.start),
        "end": _ms(s.end),
        "text": s.text,
        "avg_logprob": s.avg_logprob,
        "level_db": s.level_db,
        "words": [
            {
                "word": w.word,
                "start": _ms(w.start),
                "end": _ms(w.end),
                "score": w.score,
                "voiced": w.voiced,
                "estimated": w.estimated,
            }
            for w in s.words
        ],
    }
