"""Installation checks: one small function per item scribe needs.

``scribe check`` runs them all and prints a report; ``transcribe`` and
``master`` run the ones they need and stop at the first missing item, with
the same words and a link to the install guide section that fixes it.
"""

from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from scribe_rpg import ffmpeg
from scribe_rpg.errors import InstallError

Status = Literal["ok", "missing", "warning", "info"]

# Set at the pre-publish checklist; the repository is not public yet.
GUIDE_BASE = "https://github.com/OWNER/scribe-rpg/blob/main/docs/"
# Guide section key -> anchor on the install pages (they must provide these).
ANCHORS = {
    "ffmpeg": "ffmpeg",
    "encoder": "mp3-encoder",
    "transcription": "transcription",
    "pytorch-cuda": "pytorch-with-cuda",
    "gpu": "gpu",
    "connection": "connection-problems",
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
    return Item("ffmpeg", "ok", f"{found}  {path}")
