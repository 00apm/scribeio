"""Glossary: names for Whisper's prompt and corrections for the transcript.

Format (UTF-8 text, one entry per line)::

    # comment
    Vorgrad                    name: goes into the Whisper prompt
    Worgrad -> Vorgrad         correction, whole word or phrase
    Ostromir* -> Ostromyr      correction of a word stem; the ending is kept

Corrections match case-insensitively and insert the right side exactly as
written. The right side of a correction is also a name for the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from scribe_rpg.config import read_utf8
from scribe_rpg.errors import ScribeError


@dataclass(frozen=True)
class Correction:
    """One ``wrong -> right`` rule.

    Attributes:
        pattern: Compiled match for the wrong text. Its group ``end`` holds
            the word ending to keep (always empty for whole-word rules).
        replacement: The right text (for stem rules: the new stem).
    """

    pattern: re.Pattern[str]
    replacement: str

    def apply(self, text: str) -> str:
        """Return `text` with every match replaced."""
        # A function as replacement inserts the text literally: backslashes
        # or group references in the right side are never interpreted.
        return self.pattern.sub(lambda m: self.replacement + m.group("end"), text)


@dataclass(frozen=True)
class Glossary:
    """Parsed glossary.

    Attributes:
        names: Names for the Whisper prompt, in file order, without duplicates.
        corrections: Correction rules, in file order.
    """

    names: list[str] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)

    def correct(self, text: str) -> str:
        """Apply every correction to `text`, in file order."""
        for c in self.corrections:
            text = c.apply(text)
        return text


def load_glossary(path: Path) -> Glossary:
    """Read and parse a glossary file.

    Raises:
        ScribeError: If the file is missing, is not UTF-8 or has a malformed
            line.
    """
    path = Path(path)
    try:
        text = read_utf8(path)
    except FileNotFoundError:
        raise ScribeError(f"glossary not found: {path.name}") from None
    return parse_glossary(text, path.name)


def parse_glossary(text: str, label: str = "glossary") -> Glossary:
    """Parse glossary text.

    Args:
        text: Glossary content.
        label: File name used in error messages.

    Returns:
        The names and corrections.

    Raises:
        ScribeError: Naming the line of a malformed correction (empty side,
            more than one ``->``, or ``*`` anywhere but the end of the left
            side).
    """
    names: list[str] = []
    corrections: list[Correction] = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("->")
        if len(parts) == 1:
            _add(names, line)
            continue
        if len(parts) > 2:
            raise ScribeError(f"{label}:{n}: more than one '->'")
        wrong, right = parts[0].strip(), parts[1].strip()
        stem = wrong.endswith("*")
        if stem:
            wrong = wrong[:-1].strip()
        if not wrong or not right:
            raise ScribeError(f"{label}:{n}: empty side of '->'")
        if "*" in wrong or "*" in right:
            raise ScribeError(
                f"{label}:{n}: '*' is only allowed at the end of the left side"
            )
        corrections.append(_correction(wrong, right, stem))
        _add(names, right)
    return Glossary(names, corrections)


def _add(names: list[str], name: str) -> None:
    if name not in names:
        names.append(name)


def _correction(wrong: str, right: str, stem: bool) -> Correction:
    """Compile one rule.

    The match must start at a word boundary. A whole-word rule must also end
    at one; a stem rule swallows the rest of the word into group ``end`` so
    the ending survives (``Ostromirze`` -> ``Ostromyrze``). Spaces in the
    wrong text match any run of whitespace.
    """
    body = re.escape(wrong).replace(r"\ ", r"\s+")
    end = r"(?P<end>\w*)" if stem else r"(?P<end>)(?!\w)"
    return Correction(re.compile(rf"(?<!\w){body}{end}", re.IGNORECASE), right)
