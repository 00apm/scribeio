"""Load the built-in defaults and an optional campaign config.

One campaign config serves every command. It is sectioned like
``defaults.yaml`` and every section is optional::

    speakers:      Discord account -> name (shared by all commands)
    input:         reading Craig recordings (shared)
    transcribe:    `scribe transcribe` / `scribe merge`; also holds
                   ``glossary`` (path relative to the config file)
    master:        `scribe master` (added with audio mastering)

``speakers`` and ``transcribe.glossary`` exist only in configs; every other
key must exist in ``defaults.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from scribe_rpg.craig import sanitize
from scribe_rpg.errors import ScribeError
from scribe_rpg.master.config import merge as merge_master

# Allowed values of numeric settings: (lowest, highest, whole numbers only);
# None leaves that end open. Checked at load time, before any slow work.
_RANGES: dict[str, tuple[float | None, float | None, bool]] = {
    "transcribe.whisper.batch_size": (1, None, True),
    "transcribe.whisper.prompt_max_tokens": (0, None, True),
    "transcribe.cleanup.energy_db": (None, 0, False),
    "transcribe.cleanup.frame_ms": (1, None, False),
    "transcribe.cleanup.voiced_ratio": (0, 1, False),
    "transcribe.cleanup.phrase_max_words": (0, None, True),
    "transcribe.cleanup.loop_min_words": (0, None, True),
    "transcribe.cleanup.loop_max_repeat": (1, None, True),
    "transcribe.lines.pause": (0, None, False),
    "transcribe.bleed.similarity": (0, 1, False),
    "transcribe.bleed.min_words": (1, None, True),
    "transcribe.bleed.min_overlap": (0, 1, False),
}
# Text settings with a fixed set of values.
_CHOICES = {"transcribe.whisper.device": ("cuda", "cpu")}


@dataclass
class Config:
    """Settings for one run.

    Attributes:
        settings: Defaults merged with the config file (same keys as
            ``defaults.yaml``).
        speakers: Sanitized account name -> speaker name.
        glossary_path: Glossary file to use, if any.
    """

    settings: dict[str, Any]
    speakers: dict[str, str] = field(default_factory=dict)
    glossary_path: Path | None = None


def load_config(
    path: Path | None = None,
    glossary: Path | None = None,
    keep_fillers: bool = False,
) -> Config:
    """Build the settings for a run.

    Args:
        path: Campaign config file (``-c``), or None for defaults only.
        glossary: Glossary file from ``-g``; wins over the config's key.
        keep_fillers: ``--keep-fillers`` flag.

    Returns:
        The merged settings, speaker names and glossary path.

    Raises:
        ScribeError: If the config is missing, is not valid YAML, has an
            unknown key or a value of the wrong type.
    """
    defaults = files("scribe_rpg").joinpath("defaults.yaml").read_text(encoding="utf-8")
    settings = _parse_yaml(defaults, "defaults.yaml")
    speakers: dict[str, str] = {}
    glossary_path = None
    if path is not None:
        path = Path(path)
        user = _read_yaml(path)
        # speakers and transcribe.glossary exist only in user configs, not in
        # the defaults, so they are taken out before the unknown-key check.
        speakers = _speakers(user.pop("speakers", None) or {}, path.name)
        section = user.get("transcribe")
        if isinstance(section, dict) and (g := section.pop("glossary", None)):
            glossary_path = path.parent / str(g)
        # master has its own rules (new steps, players): see master.config
        master = user.pop("master", None)
        settings = _merge(settings, user, path.name)
        settings["master"] = merge_master(settings["master"], master, path.name)
    if glossary is not None:
        glossary_path = Path(glossary)
    if keep_fillers:
        settings["transcribe"]["cleanup"]["keep_fillers"] = True
    return Config(settings, speakers, glossary_path)


def read_utf8(path: Path) -> str:
    """Read a UTF-8 text file (a BOM is allowed).

    Raises:
        FileNotFoundError: If the file is missing.
        ScribeError: If the file is not UTF-8, naming the first bad line.
    """
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        line = data[: e.start].count(b"\n") + 1
        raise ScribeError(f"{path.name}:{line}: not UTF-8; save it as UTF-8") from None


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file; errors name the file."""
    try:
        text = read_utf8(path)
    except FileNotFoundError:
        raise ScribeError(f"config not found: {path.name}") from None
    return _parse_yaml(text, path.name)


def _parse_yaml(text: str, label: str) -> dict[str, Any]:
    """Parse YAML text that must hold key: value pairs at the top level.

    Args:
        text: YAML source.
        label: File name used in error messages.

    Raises:
        ScribeError: Naming the file and line of a YAML syntax error.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f"{label}:{mark.line + 1}" if mark else label
        problem = getattr(e, "problem", None) or e
        raise ScribeError(f"{where}: invalid YAML ({problem})") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ScribeError(f"{label}: expected key: value pairs at the top level")
    return data


def _merge(
    base: dict[str, Any], override: dict[str, Any], label: str, prefix: str = ""
) -> dict[str, Any]:
    """Merge `override` into a copy of `base`.

    Dicts merge recursively; any other value (lists included) replaces the
    default. Keys missing from `base` are errors, so typos are caught.

    Args:
        base: Default values; defines the allowed keys and their types.
        override: Values from the config file.
        label: File name used in error messages.
        prefix: Dotted path of the current section, e.g. ``"whisper."``.
    """
    out = dict(base)
    for key, value in override.items():
        name = f"{prefix}{key}"
        if key not in base:
            raise ScribeError(f"{label}: {name}: unknown key")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ScribeError(f"{label}: {name}: expected key: value pairs")
            out[key] = _merge(base[key], value, label, name + ".")
        else:
            _check_type(base[key], value, label, name)
            _check_value(value, label, name)
            out[key] = value
    return out


def _check_type(default: Any, value: Any, label: str, name: str) -> None:
    """Require `value` to have the same kind of type as the default."""
    # bool is a subclass of int in Python, so it is checked first and
    # explicitly kept out of the number case.
    if default is None:
        # Optional text setting (e.g. align_model): text, or null for "not set".
        if value is not None and not isinstance(value, str):
            raise ScribeError(f"{label}: {name}: expected text or null, got {value!r}")
        return
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, (int, float)):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif isinstance(default, list):
        ok = isinstance(value, list)
    else:
        ok = isinstance(value, str)
    if not ok:
        expected = type(default).__name__
        raise ScribeError(f"{label}: {name}: expected {expected}, got {value!r}")
    if isinstance(default, list):
        # every list setting holds text (music patterns, phrases); an empty
        # pattern would match every track
        for i, item in enumerate(value, 1):
            if not isinstance(item, str) or not item.strip():
                raise ScribeError(
                    f"{label}: {name}: item {i}: expected text, got {item!r}"
                )


def _check_value(value: Any, label: str, name: str) -> None:
    """Check a setting against its range or its set of choices, if any."""
    if name in _CHOICES and value not in _CHOICES[name]:
        allowed = " or ".join(_CHOICES[name])
        raise ScribeError(f"{label}: {name}: expected {allowed}, got {value!r}")
    if name not in _RANGES:
        return
    low, high, whole = _RANGES[name]
    if whole and not isinstance(value, int):
        raise ScribeError(f"{label}: {name}: expected a whole number, got {value!r}")
    # written so that NaN fails too
    if not ((low is None or value >= low) and (high is None or value <= high)):
        if high is None:
            limits = f"at least {low}"
        elif low is None:
            limits = f"at most {high}"
        else:
            limits = f"from {low} to {high}"
        raise ScribeError(f"{label}: {name}: expected {limits}, got {value!r}")


def _speakers(raw: Any, label: str) -> dict[str, str]:
    """Validate the speakers mapping and sanitize its account names."""
    if not isinstance(raw, dict):
        raise ScribeError(f"{label}: speakers: expected account: name pairs")
    out = {}
    keys: dict[str, str] = {}  # sanitized account -> key as written
    for account, name in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ScribeError(f"{label}: speakers: {account}: name missing")
        key = sanitize(str(account))
        if key in keys:
            raise ScribeError(
                f"{label}: speakers: {keys[key]!r} and {str(account)!r} are one "
                f"account ({key}); keep one"
            )
        keys[key] = str(account)
        out[key] = name.strip()
    return out
