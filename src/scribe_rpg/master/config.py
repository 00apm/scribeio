"""The ``master:`` section resolved into chains of steps.

A stage (``voice``, ``music``, ``mix``, or a player block over ``voice``)
holds ``chain`` (step names in order) and the steps, each ``name: {type,
options}``. Only steps named in the chain run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scribe_rpg import craig
from scribe_rpg.errors import ScribeError
from scribe_rpg.master.catalogue import Step

BITRATES = {"mp3": "96k", "m4a": "64k", "opus": "48k"}


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
    """Build the chains from ``settings["master"]`` (defaults already merged).

    Raises:
        ScribeError: If a chain names a step that is not defined.
    """
    voice = section["voice"]
    players = {
        craig.sanitize(str(account)): _chain(
            _merged(voice, block), f"master.players.{account}"
        )
        for account, block in (section.get("players") or {}).items()
    }
    fmt = section["format"]
    return MasterConfig(
        channels=section["channels"],
        format=fmt,
        bitrate=section["bitrate"] or BITRATES[fmt],
        voice=_chain(voice, "master.voice"),
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


def _merged(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Maps merge, anything else replaces (one level of steps deep)."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            out[key] = {**base[key], **value}
        else:
            out[key] = value
    return out
