"""Open a Craig recording, read its info.txt and identify the tracks.

A recording is either the zip Craig produces or a folder of processed tracks:

    session.zip                      folder/
      info.txt                         info.txt
      <id>.aup        (ignored)        <n>-<account>.flac
      <id>_data/<n>-<account>.flac
      raw.dat         (ignored)

A zip is extracted to a temporary folder that is deleted when the
`open_recording` context ends, also on errors and Ctrl+C.
"""

from __future__ import annotations

import re
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from scribe_rpg.errors import ScribeError

TRACK_FILE = re.compile(r"^(\d+)-(.+)\.flac$")
ZIP_TRACK = re.compile(r"^[^/]+_data/[^/]+\.flac$")


def sanitize(name: str) -> str:
    """Apply Craig's file-name rule to a Discord account name.

    Args:
        name: Account name as Discord shows it, e.g. ``gm.account``.

    Returns:
        The name with every character outside ``[A-Za-z0-9_-]`` replaced by
        ``_``, e.g. ``gm_account``.
    """
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def is_music(account: str, patterns: list[str]) -> bool:
    """Tell whether a track belongs to a music bot.

    Args:
        account: Account name from the track file name.
        patterns: Substrings that mark music bots (``music_patterns``).

    Returns:
        True if any pattern occurs in the account name, ignoring case.
    """
    low = account.lower()
    return any(p.lower() in low for p in patterns)


@dataclass(frozen=True)
class Info:
    """The parts of info.txt scribe needs.

    Attributes:
        start: Recording start in the system's local time zone.
        accounts: Track list, sanitized like the track file names.
        names: The same tracks as Discord shows the accounts (no ID, no ``#0``).
    """

    start: datetime
    accounts: list[str]
    names: list[str]


@dataclass(frozen=True)
class Track:
    """One audio track of a recording.

    Attributes:
        account: Account name as in the file name (already sanitized by Craig).
        kind: ``"player"`` or ``"music"``.
        path: The FLAC file.
    """

    account: str
    kind: str
    path: Path


@dataclass(frozen=True)
class Recording:
    """An opened recording.

    Attributes:
        source: Zip or folder name only, never a full path.
        start: Recording start in the system's local time zone.
        tracks: All tracks, ordered by their number in the file name.
    """

    source: str
    start: datetime
    tracks: list[Track]

    @property
    def players(self) -> list[Track]:
        """The tracks to transcribe (everything that is not music)."""
        return [t for t in self.tracks if t.kind == "player"]


def parse_info(text: str) -> Info:
    """Read the start time and track list from Craig's info.txt.

    Args:
        text: Content of info.txt.

    Returns:
        The start time (local) and the track accounts, sanitized and as
        Discord shows them.

    Raises:
        ScribeError: If the start time or the track list is missing or
            unreadable.
    """
    start = None
    names: list[str] = []
    in_tracks = False
    # info.txt is "Key:<tabs>value" lines; the track list follows "Tracks:"
    # as indented lines and ends at the first line that is not indented.
    for line in text.splitlines():
        if not line.strip():
            continue
        if in_tracks and line[:1].isspace():
            names.append(_display_name(line))
            continue
        in_tracks = False
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key == "start time":
            start = _parse_time(value.strip())
        elif key == "tracks":
            in_tracks = True
    if start is None:
        raise ScribeError("info.txt: no 'Start time' line")
    if not names:
        raise ScribeError("info.txt: no tracks listed")
    return Info(start, [sanitize(n) for n in names], names)


def _display_name(entry: str) -> str:
    """Turn a track list entry into the account as Discord shows it.

    ``gm.account#0 (1234)`` becomes ``gm.account``: the Discord ID in
    parentheses and the ``#0`` of new-style usernames are dropped, legacy
    discriminators such as ``#1234`` stay. `sanitize` then gives the form
    used in track file names.
    """
    name = re.sub(r"\s*\(\d+\)\s*$", "", entry.strip())
    return re.sub(r"#0$", "", name)


def read_info(path: Path) -> Info:
    """Read info.txt of a Craig zip or track folder, without extracting tracks.

    Raises:
        ScribeError: If the input is not a Craig recording.
    """
    path = Path(path)
    if path.is_dir():
        info_path = path / "info.txt"
        if not info_path.is_file():
            raise ScribeError(f"{path.name}: info.txt not found")
        return parse_info(info_path.read_text(encoding="utf-8-sig"))
    if not zipfile.is_zipfile(path):
        raise ScribeError(f"{path.name}: not a zip file or a folder of tracks")
    with zipfile.ZipFile(path) as zf:
        if "info.txt" not in zf.namelist():
            raise ScribeError(f"{path.name}: not a Craig recording (no info.txt)")
        return parse_info(zf.read("info.txt").decode("utf-8-sig"))


def _parse_time(value: str) -> datetime:
    """Parse Craig's UTC timestamp and convert it to local time."""
    try:
        # Python 3.10 cannot parse a trailing "Z".
        t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ScribeError(f"info.txt: cannot read start time {value!r}") from None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone()


@contextmanager
def open_recording(path: Path, music_patterns: list[str]) -> Iterator[Recording]:
    """Open a Craig zip or a folder of tracks.

    Args:
        path: The Craig zip, or a folder with info.txt and
            ``<n>-<account>.flac`` files.
        music_patterns: Substrings that mark music bot tracks.

    Yields:
        The recording. Track paths stay valid until the context ends.

    Raises:
        ScribeError: If the input is not a Craig recording, or (folder only)
            a player track listed in info.txt has no file.
    """
    path = Path(path)
    if path.is_dir():
        yield _read(
            path / "info.txt", path, path.resolve().name, music_patterns, complete=True
        )
        return
    if not zipfile.is_zipfile(path):
        raise ScribeError(f"{path.name}: not a zip file or a folder of tracks")
    # The temp folder is removed when the with-block exits, whatever the reason.
    with tempfile.TemporaryDirectory(
        prefix="scribe-", ignore_cleanup_errors=True
    ) as tmp:
        root = Path(tmp)
        data_dir = _extract(path, root)
        yield _read(
            root / "info.txt", data_dir, path.name, music_patterns, complete=False
        )


def _extract(zip_path: Path, dest: Path) -> Path:
    """Extract info.txt and the FLAC tracks; return the ``*_data`` folder.

    The .aup project and raw.dat (raw data of every track) are skipped.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n == "info.txt" or ZIP_TRACK.match(n)]
        if "info.txt" not in names:
            raise ScribeError(f"{zip_path.name}: not a Craig recording (no info.txt)")
        for name in names:
            zf.extract(name, dest)
    data_dirs = sorted(p for p in dest.glob("*_data") if p.is_dir())
    if len(data_dirs) != 1:
        raise ScribeError(
            f"{zip_path.name}: expected one *_data folder with FLAC tracks"
        )
    return data_dirs[0]


def _read(
    info_path: Path,
    track_dir: Path,
    source: str,
    music_patterns: list[str],
    complete: bool,
) -> Recording:
    """Build a Recording from info.txt and the track files in `track_dir`.

    Args:
        info_path: The info.txt file.
        track_dir: Folder holding the ``<n>-<account>.flac`` files.
        source: Name shown in outputs (zip or folder name).
        music_patterns: Substrings that mark music bot tracks.
        complete: Require a file for every player listed in info.txt (folder
            input; a missing music track is fine).
    """
    if not info_path.is_file():
        raise ScribeError(f"{source}: info.txt not found")
    info = parse_info(info_path.read_text(encoding="utf-8-sig"))
    found = []
    for f in track_dir.iterdir():
        m = TRACK_FILE.match(f.name)
        if m:
            found.append((int(m.group(1)), m.group(2), f))
    # Sort by the track number, not the name: "10-x" comes after "2-y".
    tracks = [
        Track(account, "music" if is_music(account, music_patterns) else "player", f)
        for _, account, f in sorted(found)
    ]
    if complete:
        present = {t.account for t in tracks}
        missing = [
            a
            for a in info.accounts
            if a not in present and not is_music(a, music_patterns)
        ]
        if missing:
            raise ScribeError(
                f"{source}: missing player track for {', '.join(missing)}"
            )
    if not any(t.kind == "player" for t in tracks):
        raise ScribeError(f"{source}: no player tracks found")
    return Recording(source, info.start, tracks)
