"""Create ``scribe.yaml`` and ``glossary.txt`` from the built-in templates.

The templates live in the package (``scribe_rpg/templates``), so
`scribe init` works after a plain ``pip install``. Given a recording (or a
``<name>.json``), the ``speakers`` section is prefilled with the accounts as
Discord shows them; the user only writes the names next to them. The config
holds the sections one use needs (``only``): short by default, or every
setting copied from ``defaults.yaml`` (``full``).
"""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml

from scribe_rpg import craig
from scribe_rpg.errors import ScribeError

CONFIG_NAME = "scribe.yaml"
GLOSSARY_NAME = "glossary.txt"

# The speakers block of the template: "speakers:" and its indented lines.
_SPEAKERS = re.compile(r"^speakers:\n(?:  .*\n)+", re.MULTILINE)
_LANGUAGE = re.compile(r"^(    language: )\S+", re.MULTILINE)
_KEY = re.compile(r"^([A-Za-z_]+):")

Only = Literal["master", "transcribe"] | None
# The config sections each use reads, in file order (None: both).
_SECTIONS: dict[Only, tuple[str, ...]] = {
    None: ("speakers", "input", "master", "transcribe"),
    "master": ("input", "master"),
    "transcribe": ("speakers", "input", "transcribe"),
}
_FULL_NOTE = """\
# Written with --full: every setting with its default value. A key you keep
# wins over scribe's default, also a changed one in a later version; delete
# what you do not change.

"""
_GLOSSARY = (
    "  glossary: glossary.txt    # names for Whisper + corrections; "
    "relative to this file\n"
)


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
    only: Only = None,
    full: bool = False,
) -> str:
    """The config template, optionally with real accounts and a language.

    Args:
        speakers: Accounts to list under ``speakers`` (each mapped to itself,
            for the user to rename); None keeps the template's examples.
        music: Music bot accounts, listed as a comment for reference.
        language: Value for ``transcribe.whisper.language``.
        only: ``master`` or ``transcribe`` keeps only that command's
            sections; None keeps both.
        full: Every setting with its default (``defaults.yaml``) instead of
            the short template's usual keys.
    """
    header, sections = _split(template(CONFIG_NAME))
    if full:
        header += _FULL_NOTE
        defaults = files("scribe_rpg").joinpath("defaults.yaml")
        _, built_in = _split(defaults.read_text(encoding="utf-8"))
        sections.update({k: built_in[k] for k in ("input", "master", "transcribe")})
        sections["transcribe"] = sections["transcribe"].replace(
            "transcribe:\n", "transcribe:\n" + _GLOSSARY, 1
        )
        sections["master"] = _comment_unused(sections["master"])
    if speakers is not None:
        lines = ["speakers:"]
        lines += [
            f"  {_q(name)}: {_q(name)}   # <- name in the transcript"
            for name in speakers
        ]
        if music:
            lines.append(f"  # music bots (not transcribed): {', '.join(music)}")
        # A function as replacement: names are inserted literally.
        sections["speakers"] = _SPEAKERS.sub(
            lambda _: "\n".join(lines) + "\n", sections["speakers"], count=1
        )
        if only == "master":  # no speakers section: the accounts, for `players`
            accounts = ", ".join(_q(name) for name in speakers)
            line = f"  # accounts in this recording: {accounts}\n"
            sections["master"] = sections["master"].replace(
                "master:\n", "master:\n" + line, 1
            )
    if language:
        sections["transcribe"] = _LANGUAGE.sub(
            lambda m: m.group(1) + language, sections["transcribe"], count=1
        )
    return header + "\n".join(sections[key] for key in _SECTIONS[only])


def _comment_unused(master: str) -> str:
    """Comment out the voice steps in no chain.

    They are built in, and a config may define only steps a chain names.
    """
    voice = yaml.safe_load(master)["master"]["voice"]
    for name in voice:
        if name not in [*voice["chain"], "chain", "stereo"]:
            step = re.compile(rf"^    ({re.escape(name)}:)", re.MULTILINE)
            master = step.sub(r"    # \1", master, count=1)
    return master


def _split(text: str) -> tuple[str, dict[str, str]]:
    """Split YAML text into its header and top-level sections.

    A section is a top-level key, its indented lines and the comment lines
    right above it; each ends with one newline (blank lines between them go).
    """
    lines = text.splitlines(keepends=True)
    starts = []
    for i, line in enumerate(lines):
        if _KEY.match(line):
            while i > 0 and lines[i - 1].startswith("#"):
                i -= 1
            starts.append(i)
    sections = {}
    for start, end in zip(starts, [*starts[1:], len(lines)], strict=True):
        block = lines[start:end]
        key = next(m.group(1) for line in block if (m := _KEY.match(line)))
        sections[key] = "".join(block).rstrip("\n") + "\n"
    return "".join(lines[: starts[0]]), sections


def write(
    folder: Path,
    speakers: list[str] | None = None,
    music: list[str] | None = None,
    language: str | None = None,
    force: bool = False,
    only: Only = None,
    full: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Write ``scribe.yaml`` and ``glossary.txt`` into `folder`.

    An existing glossary is kept (unless `force`), since it may hold work.
    With ``only="master"`` there is no glossary. `only` and `full` as in
    `render_config`.

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
        render_config(speakers, music, language, only, full),
        encoding="utf-8",
        newline="\n",
    )
    written, kept = [config], []
    if only == "master":
        pass  # mastering reads no glossary
    elif glossary.exists() and not force:
        kept.append(glossary)
    else:
        glossary.write_text(template(GLOSSARY_NAME), encoding="utf-8", newline="\n")
        written.append(glossary)
    return written, kept


def _q(value: str) -> str:
    """Quote a YAML key or value (a JSON string is valid YAML)."""
    return json.dumps(value, ensure_ascii=False)
