"""The ``master:`` section resolved into chains of steps.

A stage (``voice``, ``music``, ``mix``, or a player block over ``voice``)
holds ``chain`` (step names in order) and the steps, each ``name: {type,
options}``. Only steps named in the chain run.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from scribe_rpg import craig
from scribe_rpg.errors import ScribeError
from scribe_rpg.master.catalogue import FILTERS, OPERATIONS, WRITES_FILES, Step

BITRATES = {"mp3": "96k", "m4a": "64k", "opus": "48k"}
CHANNELS = ("mono", "stereo")
STAGES = ("voice", "music", "mix")
_BITRATE = re.compile(r"^\d+k$")

Fail = Callable[[str, str], ScribeError]  # (config path, problem) -> error


@dataclass(frozen=True)
class MasterConfig:
    """Resolved mastering settings.

    Attributes:
        channels: ``mono`` or ``stereo``.
        format: ``mp3``, ``m4a`` or ``opus``.
        bitrate: e.g. ``96k``; the format's default when the config says null.
        voice: Chain for player tracks without their own block.
        music: Chain for music bot tracks.
        mix: Chain for the mix of all tracks.
        players: Sanitized account -> chain (player block merged over voice).
    """

    channels: str
    format: str
    bitrate: str
    voice: list[Step]
    music: list[Step]
    mix: list[Step]
    players: dict[str, list[Step]]

    def chain_for(self, track: craig.Track) -> list[Step]:
        """The chain that processes `track`."""
        if track.kind == "music":
            return self.music
        return self.players.get(craig.sanitize(track.account), self.voice)


def resolve(section: dict[str, Any]) -> MasterConfig:
    """Build the chains from ``settings["master"]`` (output of `merge`).

    Raises:
        ScribeError: If a chain names a step that is not defined.
    """
    players = {
        craig.sanitize(str(account)): _chain(block, f"master.players.{account}")
        for account, block in (section.get("players") or {}).items()
    }
    fmt = section["format"]
    return MasterConfig(
        channels=section["channels"],
        format=fmt,
        bitrate=section["bitrate"] or BITRATES[fmt],
        voice=_chain(section["voice"], "master.voice"),
        music=_chain(section["music"], "master.music"),
        mix=_chain(section["mix"], "master.mix"),
        players=players,
    )


def _chain(stage: dict[str, Any], where: str) -> list[Step]:
    """The steps named in ``stage["chain"]``, in order."""
    steps = []
    for name in stage["chain"]:
        settings = stage.get(name)
        if not isinstance(settings, dict) or name == "chain":
            raise ScribeError(f"{where}.chain: unknown step {name!r}")
        options = {k: v for k, v in settings.items() if k != "type"}
        steps.append(Step(name, settings["type"], options, f"{where}.{name}"))
    return steps


def unknown_players(cfg: MasterConfig, accounts: list[str]) -> list[str]:
    """Player blocks (sanitized keys) for accounts not in `accounts`."""
    present = {craig.sanitize(a) for a in accounts}
    return [key for key in cfg.players if key not in present]


def merge(defaults: dict[str, Any], user: Any, label: str) -> dict[str, Any]:
    """Merge the user's ``master:`` over the defaults and validate it.

    Maps merge, lists replace: defaults -> user -> each ``players.<account>``
    over ``voice``. The returned ``players`` blocks are complete stages.

    Args:
        defaults: ``master`` from defaults.yaml.
        user: The config's ``master:`` value (None: not given).
        label: Config file name for messages.

    Raises:
        ScribeError: ``<label>: <path>: <problem>`` for the first problem.
    """

    def fail(path: str, problem: str) -> ScribeError:
        return ScribeError(f"{label}: {path}: {problem}")

    user = _map(user or {}, "master", fail)
    out = dict(defaults)
    for key, value in user.items():
        if key not in defaults:
            raise fail(f"master.{key}", "unknown key")
        out[key] = value
    if out["channels"] not in CHANNELS:
        raise fail("master.channels", "expected mono or stereo")
    if out["format"] not in BITRATES:
        raise fail("master.format", "expected mp3, m4a or opus")
    rate = out["bitrate"]
    if rate is not None and not (isinstance(rate, str) and _BITRATE.match(rate)):
        raise fail("master.bitrate", "expected e.g. 96k (or null)")
    for stage in STAGES:
        given = _map(user.get(stage) or {}, f"master.{stage}", fail)
        out[stage] = _merge_stage(defaults[stage], given, f"master.{stage}", fail)
    blocks = _map(user.get("players") or {}, "master.players", fail)
    out["players"] = {
        account: _merge_stage(
            out["voice"],
            _map(block, f"master.players.{account}", fail),
            f"master.players.{account}",
            fail,
        )
        for account, block in blocks.items()
    }
    _check_used(user, out, fail)
    return out


def _map(value: Any, path: str, fail: Fail) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise fail(path, "expected key: value pairs")
    return value


def _merge_stage(
    base: dict[str, Any], over: dict[str, Any], where: str, fail: Fail
) -> dict[str, Any]:
    """One stage: user steps merged over `base`'s; chain names checked."""
    out = dict(base)
    for name, settings in over.items():
        path = f"{where}.{name}"
        if name == "chain":
            if isinstance(settings, dict):
                raise fail(path, "'chain' is reserved and cannot be a step name")
            if not isinstance(settings, list) or not all(
                isinstance(n, str) for n in settings
            ):
                raise fail(path, "expected a list of step names")
            out["chain"] = settings
            continue
        settings = _map(settings, path, fail)
        old = base.get(name)
        kind = settings.get("type", old["type"] if old else None)
        if kind is None:
            raise fail(path, "a new step needs a type")
        if old and kind != old["type"]:
            raise fail(path, "a tweak cannot change `type`; use a new step name")
        if kind not in FILTERS | OPERATIONS:
            raise fail(path, f"unknown type {kind!r}")
        for option in WRITES_FILES.get(kind, ()):
            if option in settings:
                raise fail(path, f"option {option!r} is not allowed")
        out[name] = {**(old or {}), **settings}
    for name in out["chain"]:
        if name == "chain" or not isinstance(out.get(name), dict):
            raise fail(f"{where}.chain", f"unknown step {name!r}")
    return out


def _check_used(user: dict[str, Any], out: dict[str, Any], fail: Fail) -> None:
    """Every step the user defines is named by a chain that can see it."""
    players = out["players"]
    seen = {
        "voice": {
            *out["voice"]["chain"],
            *(n for b in players.values() for n in b["chain"]),
        },
        "music": set(out["music"]["chain"]),
        "mix": set(out["mix"]["chain"]),
    }
    stages = [(f"master.{s}", user.get(s) or {}, seen[s]) for s in STAGES]
    stages += [
        (f"master.players.{a}", block, set(players[a]["chain"]))
        for a, block in (user.get("players") or {}).items()
    ]
    for where, given, used in stages:
        for name in given:
            if name != "chain" and name not in used:
                raise fail(f"{where}.{name}", "step is in no chain")
