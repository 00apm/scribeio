"""The ``master:`` section resolved into chains of steps.

A stage (``voice``, ``music``, ``mix``, or a player block over ``voice``)
holds ``chain`` (step names in order) and the steps, each ``name: {type,
options}``. Only steps named in the chain run.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from scribeio import craig
from scribeio.errors import ScribeError
from scribeio.master.catalogue import FILTERS, OPERATIONS, WRITES_FILES, Step

BITRATES = {"mp3": "96k", "m4a": "64k", "opus": "48k"}
CHANNELS = ("mono", "stereo")
STAGES = ("voice", "music", "mix")
_BITRATE = re.compile(r"^\d+k$")
# An option name goes into the filter text as is (only values are quoted), so
# it must be a plain name: anything else could add filters to the graph.
_OPTION = re.compile(r"^[a-z][a-z0-9_]*$")

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
        voice_stereo: ``voice.stereo``: voices recorded in stereo stay stereo.
        players_stereo: Sanitized account -> its block's ``stereo``.
    """

    channels: str
    format: str
    bitrate: str
    voice: list[Step]
    music: list[Step]
    mix: list[Step]
    players: dict[str, list[Step]]
    voice_stereo: bool = False
    players_stereo: dict[str, bool] = field(default_factory=dict)

    def chain_for(self, track: craig.Track) -> list[Step]:
        """The chain that processes `track`."""
        if track.kind == "music":
            return self.music
        return self.players.get(craig.sanitize(track.account), self.voice)

    def keeps_stereo(self, track: craig.Track) -> bool:
        """Whether `track` is processed in stereo when the output is stereo.

        Music always is; a voice only if its config says ``stereo: true``.
        """
        if track.kind == "music":
            return True
        account = craig.sanitize(track.account)
        return self.players_stereo.get(account, self.voice_stereo)


def resolve(section: dict[str, Any]) -> MasterConfig:
    """Build the chains from ``settings["master"]`` (output of `merge`).

    Raises:
        ScribeError: If a chain names a step that is not defined.
    """
    blocks = section.get("players") or {}
    players = {
        craig.sanitize(str(account)): _chain(block, f"master.players.{account}")
        for account, block in blocks.items()
    }
    voice_stereo = bool(section["voice"].get("stereo", False))
    fmt = section["format"]
    return MasterConfig(
        channels=section["channels"],
        format=fmt,
        bitrate=section["bitrate"] or BITRATES[fmt],
        voice=_chain(section["voice"], "master.voice"),
        music=_chain(section["music"], "master.music"),
        mix=_chain(section["mix"], "master.mix"),
        players=players,
        voice_stereo=voice_stereo,
        players_stereo={
            craig.sanitize(str(account)): bool(block.get("stereo", voice_stereo))
            for account, block in blocks.items()
        },
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
    _unique_accounts(blocks, fail)
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
        if name == "stereo":
            if "stereo" not in base:
                raise fail(path, "only voice and players have `stereo`")
            if not isinstance(settings, bool):
                raise fail(path, "expected true or false")
            out["stereo"] = settings
            continue
        settings = _map(settings, path, fail)
        old = base.get(name)
        kind = settings.get("type", old["type"] if old else None)
        if kind is None:
            raise fail(path, "a new step needs a type")
        if old and kind != old["type"]:
            raise fail(path, "a tweak cannot change `type`; use a new step name")
        if not isinstance(kind, str) or kind not in FILTERS | OPERATIONS:
            raise fail(path, f"unknown type {kind!r}")
        for option in settings:
            if option != "type" and not (
                isinstance(option, str) and _OPTION.match(option)
            ):
                raise fail(path, f"bad option name {str(option)!r}")
        for option in WRITES_FILES.get(kind, ()):
            if option in settings:
                raise fail(path, f"option {option!r} is not allowed")
        out[name] = {**(old or {}), **settings}
        if kind == "loudness":
            _check_loudness(out[name], path, fail)
    for name in out["chain"]:
        if name == "chain" or not isinstance(out.get(name), dict):
            raise fail(f"{where}.chain", f"unknown step {name!r}")
    return out


def _check_loudness(step: dict[str, Any], path: str, fail: Fail) -> None:
    """A loudness step: a finite `target`, optional `dual_mono`, nothing else.

    It is not an ffmpeg filter, so ffmpeg never sees a typo in it.
    """
    for option in step:
        if option not in ("type", "target", "dual_mono"):
            raise fail(path, f"loudness has only target and dual_mono, not {option!r}")
    target = step.get("target")
    if (
        not isinstance(target, (int, float))
        or isinstance(target, bool)
        or not math.isfinite(target)
    ):
        raise fail(path, "loudness needs a target in LUFS, e.g. -19")
    if not -70 <= target <= 0:  # e.g. -190 for -19: a nearly silent master
        raise fail(path, f"target: expected from -70 to 0 LUFS, got {target!r}")
    if not isinstance(step.get("dual_mono", True), bool):
        raise fail(path, "dual_mono: expected true or false")


def _unique_accounts(blocks: dict[str, Any], fail: Fail) -> None:
    """Two ``players`` keys must not name one (sanitized) account."""
    seen: dict[str, str] = {}
    for account in blocks:
        key = craig.sanitize(str(account))
        if key in seen:
            raise fail(
                "master.players",
                f"{seen[key]!r} and {str(account)!r} are one account ({key}); keep one",
            )
        seen[key] = str(account)


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
            if name not in ("chain", "stereo") and name not in used:
                raise fail(f"{where}.{name}", "step is in no chain")
