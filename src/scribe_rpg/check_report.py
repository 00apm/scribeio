"""The `scribe check` report: what scribe needs, what it finds, per use.

Offline: it reads the local caches only, unless asked to look up model
sizes and test the model sources (``--online``). The settings checked are the
``scribe.yaml`` given with ``-c``, else the one in the current folder, else
the defaults.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scribe_rpg import __version__, checks, init
from scribe_rpg.checks import Item
from scribe_rpg.config import load_config
from scribe_rpg.errors import ScribeError

if TYPE_CHECKING:
    from scribe_rpg.transcribe.models import Model

SYMBOLS = {"ok": "✔", "missing": "✘", "warning": "⚠", "info": "ℹ"}
NOT_INSTALLED = "not installed (audio cleanup only)"
NONE_YET = "none yet: downloaded on your first transcription"
# Model source -> the host its downloads come from (checks.SOURCES)
SOURCE_HOSTS = {"huggingface": "huggingface.co", "torchaudio": "download.pytorch.org"}


@dataclass
class Checked:
    """The settings the report checks.

    Attributes:
        path: The config file, or None when there is none.
        settings: Its settings; the defaults without a config or when it
            does not load.
        error: Why the config does not load, or ``""``.
    """

    path: Path | None
    settings: dict[str, Any]
    error: str = ""


@dataclass
class Group:
    """One use (audio cleanup, transcription): its verdict and items."""

    title: str
    verdict: str
    items: list[Item]


def load_settings(config: Path | None, cwd: Path) -> Checked:
    """The config from ``-c``, else ``scribe.yaml`` in `cwd`, else defaults."""
    path = config or init.find_config(cwd)
    if path is None:
        return Checked(None, load_config().settings)
    try:
        return Checked(path, load_config(path).settings)
    except ScribeError as e:
        return Checked(path, load_config().settings, str(e))


def header() -> str:
    """scribe and Python versions, and how scribe was installed."""
    route = "installed with uv" if checks.uv_install() else f"in {sys.prefix}"
    return f"scribe {__version__}, Python {platform.python_version()} ({route})"


def audio_group(settings: dict[str, Any]) -> Group:
    """ffmpeg, and the encoder of the configured output format."""
    from scribe_rpg.master.config import BITRATES

    found = checks.ffmpeg_item()
    items = [found]
    if found.status != "missing":
        fmt = settings["master"]["format"]
        bitrate = settings["master"]["bitrate"] or BITRATES[fmt]
        items.append(checks.encoder_item(fmt, bitrate))
    return Group("Audio cleanup", _verdict(items), items)


def transcription_group(settings: dict[str, Any], ffmpeg: Item) -> Group:
    """WhisperX, PyTorch and the GPU for the configured device.

    Args:
        settings: The checked settings (``transcribe.whisper.device``).
        ffmpeg: The ffmpeg item: transcription decodes audio with ffmpeg.
    """
    wx = checks.whisperx_item()
    if wx.message == checks.WHISPERX_NOT_INSTALLED:
        return Group("Transcription", NOT_INSTALLED, [wx])
    items = [wx]
    if wx.status != "missing":
        device = settings["transcribe"]["whisper"]["device"]
        torch = checks.torch_item(device)
        items.append(torch)
        # with a CPU-only PyTorch the GPU's reason would repeat PyTorch's
        if torch.status != "missing":
            items.append(checks.gpu_item(device))
        if ffmpeg.status != "missing":
            # the detail starts with the version ("9.0.2  C:\...\ffmpeg.exe")
            info = checks.audio_library_item(ffmpeg.detail.split(" ")[0])
            if info is not None:
                items.append(info)
    verdict = "not ready" if ffmpeg.status == "missing" else _verdict(items)
    return Group("Transcription", verdict, items)


def _verdict(items: list[Item]) -> str:
    return "not ready" if any(i.status == "missing" for i in items) else "ready"


def group_lines(group: Group) -> list[str]:
    """The title line with the verdict, then one line per item."""
    lines = [f"{group.title:<53}{group.verdict}"]
    if group.verdict == NOT_INSTALLED:
        [wx] = group.items
        return [
            *lines,
            "  " + SYMBOLS["info"] + " " + checks.line(wx),
            "    see " + checks.guide_url("transcription"),
        ]
    for item in group.items:
        lines.append("  " + SYMBOLS[item.status] + " " + checks.line(item))
        if item.status == "missing":
            lines.append("    see " + checks.guide_url(item.guide))
    return lines


def needed_models(checked: Checked, whisperx: bool) -> tuple[list[Model], str]:
    """The models the checked config needs, or why they cannot be named.

    Returns:
        The models and ``""``; no models without a config that loads or
        without WhisperX; no models and the error for an unknown model.
    """
    from scribe_rpg.transcribe import models

    if checked.path is None or checked.error or not whisperx:
        return [], ""
    try:
        return models.needed(checked.settings["transcribe"]["whisper"]), ""
    except ScribeError as e:
        return [], str(e)


def model_lines(
    checked: Checked,
    cwd: Path,
    whisperx: bool,
    needed: tuple[list[Model], str],
    online: bool = False,
    reachable: frozenset[str] = frozenset(),
) -> list[str]:
    """The models on disk (with WhisperX loaded) and the settings checked.

    Args:
        checked: The settings checked.
        cwd: The current folder, named when it holds no config.
        whisperx: True if WhisperX loaded; models are listed only then.
        needed: `needed_models` for these settings.
        online: Add the download size of the missing models.
        reachable: Model sources that answered; sizes are looked up only
            there (a dropped connection would stall each lookup).
    """
    from scribe_rpg.transcribe import models

    lines = []
    if whisperx:
        ready, damaged = models.local_alignment()
        lines += [
            f"Models ({models.cache_folder()})",
            "  Whisper:   " + (", ".join(models.cached_whisper_models()) or NONE_YET),
            "  Alignment: " + (", ".join(ready) or NONE_YET),
        ]
        lines += [
            f"  ⚠ alignment {language}: damaged; replaced on the next transcription"
            for language in damaged
        ]
    else:
        lines.append("Settings")  # no Models block to hold the settings lines
    if checked.path is None:
        return [
            *lines,
            f"  No {init.CONFIG_NAME} in {cwd} (the folder this terminal is in).",
            "  To set up a session: go to the folder with your recording and run "
            "scribe init",
        ]
    # absolute(), not resolve(): a mapped drive (Z:\) would become \\server\...
    path = checked.path.absolute()
    found, error = needed
    error = checked.error or error
    if error:
        # config errors start with the file name ("scribe.yaml:3: ..."):
        # the full path takes its place, so the name is not said twice
        if error.startswith(path.name):
            text = str(path) + error.removeprefix(path.name)
        else:
            text = f"{path}: {error}"
        return [*lines, f"  ⚠ Settings checked: {text}"]
    lines.append(f"  Settings checked: {path}")
    if not found:  # without WhisperX the models cannot be named
        return lines

    def need(m: Model) -> str:
        name = f"alignment {m.language}" if m.kind == "alignment model" else m.name
        if m.cached:
            return f"{name} ✔"
        size = ""
        if online:
            nbytes = models.size(m) if SOURCE_HOSTS.get(m.source) in reachable else None
            size = f"{models.amount(nbytes)}, "
        return f"{name} ↓ ({size}downloaded on first transcription)"

    return [*lines, "    needs " + ", ".join(need(m) for m in found)]


def connection_items(needed: list[Model]) -> list[Item]:
    """Whether the model sources answer: Hugging Face always, PyTorch's
    download site when a needed alignment model comes from there."""
    hosts = ["huggingface.co"]
    if any(m.source == "torchaudio" for m in needed):
        hosts.append("download.pytorch.org")
    return [checks.reach(host, checks.SOURCES[host]) for host in hosts]


def connection_lines(items: list[Item]) -> list[str]:
    """The Connection block, with a hint when a source does not answer."""
    lines = ["Connection"]
    lines += ["  " + SYMBOLS[i.status] + " " + checks.line(i) for i in items]
    if any(i.status != "ok" for i in items):
        lines += [
            "  " + SYMBOLS["info"] + " " + checks.CONNECTION_INFO,
            "    see " + checks.guide_url("connection"),
        ]
    return lines


def report(config: Path | None, cwd: Path, online: bool) -> tuple[list[str], int]:
    """The whole report and the exit code (1 if a use is not ready).

    Args:
        config: The ``-c`` config, or None.
        cwd: The current folder (its ``scribe.yaml`` is checked without -c).
        online: Also look up missing model sizes and test the model sources;
            neither changes the exit code.
    """
    checked = load_settings(config, cwd)
    audio = audio_group(checked.settings)
    transcription = transcription_group(checked.settings, audio.items[0])
    whisperx = transcription.items[0].status != "missing"
    needed = needed_models(checked, whisperx)
    # tested first, so the size lookups skip a source that does not answer
    connection = connection_items(needed[0]) if online else []
    reachable = frozenset(i.name for i in connection if i.status == "ok")
    lines = [
        header(),
        "",
        *group_lines(audio),
        "",
        *group_lines(transcription),
        "",
        *model_lines(checked, cwd, whisperx, needed, online, reachable),
    ]
    if online:
        lines += ["", *connection_lines(connection)]
    code = 1 if "not ready" in (audio.verdict, transcription.verdict) else 0
    return lines, code
