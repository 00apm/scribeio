"""The step types a ``master:`` chain may use, and quoting of their values.

A step is either an allowed ffmpeg filter (its options are the filter's own,
passed unchanged) or a scribe operation (``loudness``). Adding a filter is one
name in FILTERS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FILTERS = frozenset(
    {
        "highpass",
        "lowpass",
        "bandreject",
        "equalizer",
        "firequalizer",
        "afftdn",
        "agate",
        "acompressor",
        "deesser",
        "volume",
        "alimiter",
    }
)
OPERATIONS = frozenset({"loudness"})
# Options that make ffmpeg write a file; never allowed in a config.
WRITES_FILES = {"firequalizer": frozenset({"dumpfile"})}


@dataclass(frozen=True)
class Step:
    """One step of a chain.

    Attributes:
        name: Step name in the config, e.g. ``comp``.
        type: An ffmpeg filter from FILTERS, or an operation from OPERATIONS.
        options: The filter's options, or the operation's settings.
        where: Config path for messages, e.g. ``master.voice.comp``.
    """

    name: str
    type: str
    options: dict[str, Any]
    where: str = ""


def quote(value: Any) -> str:
    """A filter option value, escaped so no character can break the graph.

    ffmpeg unescapes twice: first the filter graph (``[ ] , ;`` are special),
    then the filter's option list (``:`` is special); ``\\`` and ``'`` are
    special in both.
    """
    text = ("true" if value else "false") if isinstance(value, bool) else str(value)
    text = _escape(text, "\\':")
    return _escape(text, "\\'[],;")


def _escape(text: str, special: str) -> str:
    return "".join("\\" + c if c in special else c for c in text)
