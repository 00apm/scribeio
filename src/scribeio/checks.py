"""Installation checks: one small function per item scribe needs.

``scribe check`` runs them all and prints a report; ``transcribe`` and
``master`` run the ones they need and stop at the first missing item, with
the same words and a link to the install guide section that fixes it.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scribeio import ffmpeg
from scribeio.errors import InstallError

Status = Literal["ok", "missing", "warning", "info"]

GUIDE_BASE = "https://github.com/00apm/scribeio/blob/main/docs/"
# Guide section key -> anchor on the install pages (they must provide these).
ANCHORS = {
    "ffmpeg": "ffmpeg",
    "encoder": "mp3-encoder",
    "transcription": "transcription",
    "pytorch-cuda": "pytorch-with-cuda",
    "gpu": "gpu",
    "connection": "connection-problems",
    "app-control": "windows-smart-app-control",
}
CHECK_HINT = "Run `scribe check` to see everything scribe needs."


@dataclass(frozen=True)
class Item:
    """The result of one check.

    Attributes:
        name: What was checked, e.g. ``ffmpeg``, ``mp3 encoder``, ``GPU``.
        status: ``ok``, ``missing`` (that use will not work), ``warning``
            (works, with a caveat) or ``info``.
        detail: Version, path or device name.
        message: Plain-words reason; never repeats `name`.
        guide: Guide section key (``ANCHORS``), or ``""``.
    """

    name: str
    status: Status
    detail: str = ""
    message: str = ""
    guide: str = ""


def uv_install(prefix: str | None = None) -> bool:
    """True if scribe runs in a uv tool environment (holds uv-receipt.toml)."""
    return (Path(prefix or sys.prefix) / "uv-receipt.toml").is_file()


def guide_url(key: str) -> str:
    """Link to a guide section, on the install page of this install route."""
    page = "install.md" if uv_install() else "install-own-env.md"
    return f"{GUIDE_BASE}{page}#{ANCHORS[key]}"


def line(item: Item) -> str:
    """``<name> <detail>: <message>``, leaving out the empty parts."""
    text = f"{item.name} {item.detail}".rstrip()
    return f"{text}: {item.message}" if item.message else text


def fail(item: Item) -> InstallError:
    """The error for a missing item: its line, then the guide link."""
    return InstallError(f"{line(item)}\nsee {guide_url(item.guide)}")


def require(item: Item) -> Item:
    """Return `item`, or raise its InstallError if it is missing."""
    if item.status == "missing":
        raise fail(item)
    return item


def installed_version(package: str) -> str:
    """Version of an installed package, or ``"unknown"``."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def parse_version(text: str) -> tuple[int, int, int] | None:
    """``"3.8.6.post1"`` -> (3, 8, 6); None if it does not look like a version."""
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)


def ffmpeg_item() -> Item:
    """ffmpeg in PATH, with its version and path; old versions warned about.

    Git builds (``N-11234-gabc``) and an unreadable version are not warned
    about.
    """
    path = shutil.which("ffmpeg")
    if path is None:
        return Item(
            "ffmpeg", "missing",
            message="not found in PATH: scribe needs it to read and write audio",
            guide="ffmpeg",
        )  # fmt: skip
    found = ffmpeg.version()
    parsed = parse_version(found)
    if parsed is not None and parsed[:2] < ffmpeg.MIN_VERSION:
        need = ".".join(map(str, ffmpeg.MIN_VERSION))
        return Item(
            "ffmpeg", "warning", found,
            f"older than {need}, the oldest scribe supports; mastering may fail",
        )  # fmt: skip
    # Windows' which() spells the extension as PATHEXT does: ffmpeg.EXE
    root, ext = os.path.splitext(path)
    return Item("ffmpeg", "ok", f"{found}  {root}{ext.lower()}")


def encoder_item(fmt: str, bitrate: str) -> Item:
    """Encode 0.1 s of silence as `master` would encode a `fmt` file.

    Returns:
        ok; missing if this ffmpeg lacks the encoder; a warning with the
        encoder's complaint if it refuses the bitrate (a settings problem).
    """
    # imported here: the CLI imports this module for every command, and
    # mix imports this module itself
    from scribeio.master.mix import encode_args, error_lines

    name = f"{fmt} encoder"
    args = encode_args(fmt, bitrate)
    # errors only: the stream info lines would come before the reason
    try:
        done = ffmpeg.attempt(
            ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "0.1",
             "-loglevel", "error", *args, "-f", "null", "-"]
        )  # fmt: skip
    except OSError as e:  # in PATH but does not start (a broken shim, a bad file)
        return Item(
            name, "missing",
            message=f"cannot be tried: ffmpeg does not start ({_first_line(e)})",
            guide="ffmpeg",
        )  # fmt: skip
    if done.returncode == 0:
        return Item(name, "ok")
    codec = args[1]  # "-c:a", codec, ...
    if f"Unknown encoder '{codec}'" in done.stderr:
        message = f"this ffmpeg has no {codec}, so it cannot write {fmt} files"
        if fmt != "m4a":
            message += "; --format m4a works without it"
        return Item(name, "missing", message=message, guide="encoder")
    lines = error_lines(done.stderr)
    # the encoder's own complaint comes first (e.g. "The bit rate ... is
    # unsupported"); the lines after it are ffmpeg's generic follow-ups
    first = lines[0] if lines else f"ffmpeg exited with {done.returncode}"
    return Item(name, "warning", bitrate, first)


TESTED = "3.8.6"
TESTED_MIN = (3, 8, 6)  # older versions lack APIs scribe uses
TESTED_BELOW = (3, 9, 0)  # a new minor version may pin another PyTorch
WHISPERX_NOT_INSTALLED = (
    "not installed, so scribe cannot transcribe (audio mastering works)"
)
ON_CPU = "set to run on the CPU: very slow"
BLOCKED_BY_WINDOWS = (
    "Windows Smart App Control blocked one of its files, so scribe cannot "
    "transcribe on this PC (audio mastering works)"
)
# Windows: "An Application Control policy has blocked this file."
APP_CONTROL_BLOCKED = 4551


def _first_line(e: BaseException) -> str:
    text = str(e).strip()
    return text.splitlines()[0] if text else type(e).__name__


def whisperx_item() -> Item:
    """WhisperX importable, with its version; untested versions warned about.

    An install that fails to import for another reason than WhisperX itself
    missing (a DLL that does not load, a missing dependency) is broken.
    """
    # imported here: whisper imports this module
    from scribeio.transcribe.whisper import hidden_warnings

    try:
        # the known noisy warnings (torchcodec with ffmpeg 8+) stay off the
        # screen, as in transcribe; the ffmpeg item's info explains that one
        with hidden_warnings():
            import whisperx  # noqa: F401

            # WhisperX imports lazily: these load torch, torchaudio,
            # transformers and faster-whisper, where a broken install shows
            # (a few seconds)
            import whisperx.alignment
            import whisperx.asr  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name == "whisperx":
            return Item(
                "WhisperX", "missing",
                message=WHISPERX_NOT_INSTALLED, guide="transcription",
            )  # fmt: skip
        return _broken_whisperx(e)
    except Exception as e:
        return _broken_whisperx(e)
    found = installed_version("whisperx")
    parsed = parse_version(found)
    if parsed and not TESTED_MIN <= parsed < TESTED_BELOW:
        return Item(
            "WhisperX", "warning", found,
            f"outside the tested range (>=3.8.6,<3.9; tested {TESTED}); "
            "transcription may fail",
        )  # fmt: skip
    return Item("WhisperX", "ok", found)


def _blocked_by_windows(e: Exception) -> bool:
    """True if Windows Application Control (Smart App Control) refused a file."""
    if getattr(e, "winerror", None) == APP_CONTROL_BLOCKED:
        return True
    if sys.platform != "win32":
        return False
    # A blocked .pyd gives an ImportError with only Windows' text, in the
    # language of Windows: ask Windows for that text.
    import ctypes

    text = ctypes.FormatError(APP_CONTROL_BLOCKED).strip()
    return bool(text) and text in str(e)


def _broken_whisperx(e: Exception) -> Item:
    if _blocked_by_windows(e):
        return Item(
            "WhisperX", "missing",
            message=f"{BLOCKED_BY_WINDOWS} ({type(e).__name__}: {_first_line(e)})",
            guide="app-control",
        )  # fmt: skip
    return Item(
        "WhisperX", "missing",
        message=f"installed but does not load ({type(e).__name__}: {_first_line(e)})",
        guide="transcription",
    )  # fmt: skip


def torch_item(device: str) -> Item:
    """PyTorch's version and whether it is a CUDA build.

    Args:
        device: ``transcribe.whisper.device``; with ``cpu`` the item is info.
    """
    import torch

    found = installed_version("torch")
    cuda_build = torch.version.cuda is not None
    detail = f"{found}, CUDA build" if cuda_build else found
    if device == "cpu":
        return Item("PyTorch", "info", detail, ON_CPU)
    if not cuda_build:
        return Item(
            "PyTorch", "missing", found,
            "a CPU-only build, so transcription would take hours", "pytorch-cuda",
        )  # fmt: skip
    return Item("PyTorch", "ok", detail)


def gpu_item(device: str) -> Item:
    """The GPU PyTorch sees, with its memory, or why it cannot use it.

    Args:
        device: ``transcribe.whisper.device``; with ``cpu`` the item is info
            and PyTorch is not asked.
    """
    if device == "cpu":
        return Item("GPU", "info", message=ON_CPU)
    import torch

    try:
        if torch.cuda.is_available():
            # torch warns at length on its first CUDA call for a card its
            # build cannot run; the item below says it in one line
            with warnings.catch_warnings():
                for text in _OLD_GPU_WARNINGS:
                    warnings.filterwarnings("ignore", message=text)
                name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            # as the message, so the line reads "GPU: <name>, 8 GB"
            found = f"{name}, {memory:.0f} GB"
            if _too_old(torch):
                return Item(
                    "GPU", "missing",
                    message=f"{found}: the installed PyTorch does not support "
                    "this GPU; set device: cpu to transcribe on the CPU (very "
                    "slow, hours per recording)",
                    guide="gpu",
                )  # fmt: skip
            return Item("GPU", "ok", message=found)
        reason = _no_gpu_reason(torch)
    except Exception as e:
        reason = _first_line(e)
    return Item(
        "GPU", "missing",
        message=f"not seen ({reason}); transcription would take hours on the CPU",
        guide="gpu",
    )  # fmt: skip


# Starts of the warnings torch gives for a card older than its build supports
# (matched from the message's start, after its leading whitespace).
_OLD_GPU_WARNINGS = (
    r"\s*Found GPU\d+ .* which is of cuda capability",
    r"\s*Please install PyTorch with a following CUDA",
    r"\s*.* with CUDA capability sm_\d+ is not compatible",
)


def _too_old(torch: Any) -> bool:
    """The GPU is older than every architecture the PyTorch build has code for.

    Such a card is seen, but its first CUDA work fails: PyTorch 2.8's CUDA
    12.9 build starts at compute capability 7.0, so a GTX 900/1000 card
    cannot run it. A card newer than the list still runs (same-major code,
    PTX), so only "older" counts.
    """
    archs = [int(a[3:]) for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    if not archs:
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return bool(major * 10 + minor < min(archs))


def _no_gpu_reason(torch: Any) -> str:
    """Why CUDA is not available: torch's own error when it gives one."""
    if torch.version.cuda is None:
        return "PyTorch is a CPU-only build"
    try:
        torch.cuda.init()
    except Exception as e:
        return _first_line(e)
    return "no CUDA device found"


def audio_library_item(ffmpeg_version: str) -> Item | None:
    """Info about the warning WhisperX's audio library prints with ffmpeg 8+.

    Returns:
        The item for ffmpeg 8 or newer; None otherwise or for an unreadable
        version.
    """
    parsed = parse_version(ffmpeg_version)
    if parsed is None or parsed[0] < 8:
        return None
    return Item(
        "ffmpeg", "info", str(parsed[0]),
        "WhisperX's audio library prints a warning; harmless",
    )  # fmt: skip


# Where models download from; a copy of models.TORCHAUDIO_URL (models imports
# the network code, so it is not imported here).
SOURCES = {
    "huggingface.co": "https://huggingface.co",
    "download.pytorch.org": "https://download.pytorch.org/torchaudio/models/",
}
CONNECTION_INFO = "a firewall, antivirus or proxy may be blocking Python"


def reach(host: str, url: str, timeout: float = 10.0) -> Item:
    """Whether `url` answers at all; any HTTP answer, even an error, counts.

    Returns:
        ok, or a warning saying why the request got no answer.
    """
    request = Request(url, method="HEAD", headers={"User-Agent": "scribe"})
    try:
        with urlopen(request, timeout=timeout):
            pass
    except HTTPError:
        pass  # the server answered
    except URLError as e:
        return _unreachable(host, str(e.reason))
    except OSError as e:  # a timeout while reading the answer
        return _unreachable(host, str(e) or "timed out")
    except Exception as e:  # a garbled answer, a bad proxy setting
        return _unreachable(host, _first_line(e))
    return Item(host, "ok")


def _unreachable(host: str, reason: str) -> Item:
    return Item(host, "warning", message=f"not reachable: {reason}", guide="connection")
