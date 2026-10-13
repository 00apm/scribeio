"""Create ``scribe.yaml`` and ``glossary.txt`` from the built-in templates.

The templates live in the package (``scribe_rpg/templates``), so
`scribe init` works after a plain ``pip install``. Given a recording (or a
``<name>.json``), the ``speakers`` section is prefilled with the accounts as
Discord shows them; the user only writes the names next to them.
"""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path

from scribe_rpg import craig
from scribe_rpg.errors import ScribeError

CONFIG_NAME = "scribe.yaml"
GLOSSARY_NAME = "glossary.txt"

# The speakers block of the template: "speakers:" and its indented lines.
_SPEAKERS = re.compile(r"^speakers:\n(?:  .*\n)+", re.MULTILINE)
_LANGUAGE = re.compile(r"^(    language: )\S+", re.MULTILINE)


def template(name: str) -> str:
    """Text of a built-in template (``scribe.yaml`` or ``glossary.txt``)."""
    # one name per joinpath: Python 3.10 Traversable takes one
    folder = files("scribe_rpg").joinpath("templates")
    return folder.joinpath(name).read_text(encoding="utf-8")


def find_config(folder: Path) -> Path | None:
    """``scribe.yaml`` in `folder`, if there is one."""
    path = Path(folder) / CONFIG_NAME
    return path if path.is_file() else None


def speakers_from(
    source: Path, music_patterns: list[str]
) -> tuple[list[str], list[str]]:
    """Accounts of a recording or session JSON, split into players and music.

    Args:
        source: Craig zip, track folder, or a ``<name>.json`` from transcribe.
        music_patterns: Substrings that mark music bots.

    Returns:
        Player accounts and music accounts. From a zip they are written as
        Discord shows them (info.txt); a track folder and a JSON give the
        file-name form, which works as well (scribe sanitizes config keys).
    """
    source = Path(source)
    if source.suffix.lower() == ".json":
        from scribe_rpg.transcribe import session_json

        tracks = session_json.read(source).tracks
        players = [t.account for t in tracks if t.kind == "player"]
        return players, [t.account for t in tracks if t.kind == "music"]
    if source.is_dir():
        names = [craig.account_of(f) for f in craig.dir_tracks(source)]
        if not names:
            raise ScribeError(f"{source.name}: no FLAC files")
    else:
        names = craig.read_info(source).names
    music = [n for n in names if craig.is_music(n, music_patterns)]
    return [n for n in names if n not in music], music


def render_config(
    speakers: list[str] | None = None,
    music: list[str] | None = None,
    language: str | None = None,
) -> str:
    """The config template, optionally with real accounts and a language.

    Args:
        speakers: Accounts to list under ``speakers`` (each mapped to itself,
            for the user to rename); None keeps the template's examples.
        music: Music bot accounts, listed as a comment for reference.
        language: Value for ``transcribe.whisper.language``.
    """
    text = template(CONFIG_NAME)
    if speakers is not None:
        lines = ["speakers:"]
        lines += [
            f"  {_q(name)}: {_q(name)}   # <- name in the transcript"
            for name in speakers
        ]
        if music:
            lines.append(f"  # music bots (not transcribed): {', '.join(music)}")
        # A function as replacement: names are inserted literally.
        text = _SPEAKERS.sub(lambda _: "\n".join(lines) + "\n", text, count=1)
    if language:
        text = _LANGUAGE.sub(lambda m: m.group(1) + language, text, count=1)
    return text


def write(
    folder: Path,
    speakers: list[str] | None = None,
    music: list[str] | None = None,
    language: str | None = None,
    force: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Write ``scribe.yaml`` and ``glossary.txt`` into `folder`.

    An existing glossary is kept (unless `force`), since it may hold work.

    Returns:
        Files written and existing files kept.

    Raises:
        ScribeError: If ``scribe.yaml`` exists and `force` is not set.
    """
    folder = Path(folder)
    config = folder / CONFIG_NAME
    glossary = folder / GLOSSARY_NAME
    if config.exists() and not force:
        where = folder.name or "."
        raise ScribeError(
            f"{CONFIG_NAME} already exists in {where} (use --force to overwrite)"
        )
    folder.mkdir(parents=True, exist_ok=True)
    config.write_text(
        render_config(speakers, music, language), encoding="utf-8", newline="\n"
    )
    written, kept = [config], []
    if glossary.exists() and not force:
        kept.append(glossary)
    else:
        glossary.write_text(template(GLOSSARY_NAME), encoding="utf-8", newline="\n")
        written.append(glossary)
    return written, kept


def _q(value: str) -> str:
    """Quote a YAML key or value (a JSON string is valid YAML)."""
    return json.dumps(value, ensure_ascii=False)
