"""Write ``<name> - raport.md``: the quality report (in English).

Sections: versions used, per-track summary, warnings, every dropped segment
grouped by reason, and possible bleed between tracks.
"""

from __future__ import annotations

from pathlib import Path

from scribe_rpg.transcribe.markdown import duration, hms
from scribe_rpg.transcribe.pipeline import MergeResult
from scribe_rpg.transcribe.session import RawSession

REASONS = [
    ("energy", "Energy (track silent under most words)"),
    ("phrase", "Known hallucination phrase"),
    ("filler", "Hesitation filler"),
    ("loop", "Repeated loop"),
]


def render(raw: RawSession, result: MergeResult) -> str:
    """Return the report as Markdown text."""
    out = [f"# Report: {raw.source}", "", _versions(raw), ""]
    out += _summary(result)
    out += ["", "## Warnings", ""]
    out += [f"- {w}" for w in result.warnings] or ["None."]
    out += ["", "## Dropped segments", ""]
    out += _dropped(result)
    out += ["## Possible bleed", ""]
    out += _bleed(result) or ["None."]
    return "\n".join(out) + "\n"


def write(path: Path, raw: RawSession, result: MergeResult) -> None:
    """Write the report as UTF-8 with LF line endings."""
    Path(path).write_text(render(raw, result), encoding="utf-8", newline="\n")


def _versions(raw: RawSession) -> str:
    """One line such as ``scribe 0.1.0 · whisperx 3.8.6 · … · large-v3 · pl``."""
    used = {"scribe": raw.scribe, **raw.versions}
    parts = [f"{name} {v}" for name, v in used.items() if v]
    keys = ("model", "language", "device")
    parts += [str(raw.whisper[k]) for k in keys if raw.whisper.get(k)]
    return " · ".join(parts)


def _summary(result: MergeResult) -> list[str]:
    out = [
        "## Summary",
        "",
        "| account | speaker | kind | segments kept | dropped | words kept "
        "| speaking time |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for s in result.stats:
        out.append(
            f"| {s.account} | {s.speaker} | {s.kind} | {s.kept} | {s.dropped} "
            f"| {s.words} | {duration(s.speaking_time)} |"
        )
    return out


def _dropped(result: MergeResult) -> list[str]:
    """One subsection per reason, each with a count and its segments in time order."""
    out = []
    for reason, title in REASONS:
        items = sorted(
            (d for d in result.dropped if d.reason == reason),
            key=lambda d: d.segment.start,
        )
        out += [f"### {title} ({len(items)})", ""]
        for d in items:
            detail = f" ({d.detail})" if d.detail else ""
            speaker = result.speaker_of(d.segment.account)
            text = d.segment.text.strip()
            out.append(f"- [{hms(d.segment.start)}] {speaker}: {text}{detail}")
        out.append("")
    return out


def _bleed(result: MergeResult) -> list[str]:
    """Bleed pairs with both levels: the quieter side is the likely leak."""
    out = []
    for b in result.bleed:
        a_side = f"{result.speaker_of(b.a.account)} ({_db(b.a.level_db)})"
        b_side = f"{result.speaker_of(b.b.account)} ({_db(b.b.level_db)})"
        out.append(f"- [{hms(b.start)}] {a_side} / {b_side}: {b.a.text.strip()}")
    return out


def _db(level: float | None) -> str:
    return "no level" if level is None else f"{level:.0f} dB"
