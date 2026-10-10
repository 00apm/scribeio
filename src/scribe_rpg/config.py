"""Load the built-in defaults and an optional transcription config.

The config file may override any key of ``defaults.yaml`` and adds two keys
of its own: ``speakers`` (Discord account -> name in the transcript) and
``glossary`` (path to the glossary file, relative to the config).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from scribe_rpg.craig import sanitize
from scribe_rpg.errors import ScribeError


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
        path: Transcription config file (``-c``), or None for defaults only.
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
        # speakers and glossary exist only in user configs, not in the defaults.
        speakers = _speakers(user.pop("speakers", None) or {}, path.name)
        if (g := user.pop("glossary", None)) is not None:
            glossary_path = path.parent / str(g)
        settings = _merge(settings, user, path.name)
    if glossary is not None:
        glossary_path = Path(glossary)
    if keep_fillers:
        settings["filters"]["keep_fillers"] = True
    return Config(settings, speakers, glossary_path)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file; errors name the file."""
    try:
        text = path.read_text(encoding="utf-8-sig")
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
            out[key] = value
    return out


def _check_type(default: Any, value: Any, label: str, name: str) -> None:
    """Require `value` to have the same kind of type as the default."""
    # bool is a subclass of int in Python, so it is checked first and
    # explicitly kept out of the number case.
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


def _speakers(raw: Any, label: str) -> dict[str, str]:
    """Validate the speakers mapping and sanitize its account names."""
    if not isinstance(raw, dict):
        raise ScribeError(f"{label}: speakers: expected account: name pairs")
    out = {}
    for account, name in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ScribeError(f"{label}: speakers: {account}: name missing")
        out[sanitize(str(account))] = name.strip()
    return out
