"""Open a Craig recording, read its info.txt and identify the tracks.

A recording is either the zip Craig produces or a folder of tracks:

    session.zip                      folder/
      info.txt                         info.txt        (optional)
      <id>.aup        (ignored)        <n>-<account>.flac or <account>.flac
      <id>_data/<n>-<account>.flac
      raw.dat         (ignored)

Craig's FLAC download has the tracks at the top of the zip, without the
project file and the ``*_data`` folder; both layouts are read.

In a folder every FLAC is one track; its account is the file stem without a
leading ``<n>-``. A folder with no FLACs but one ``*_data`` folder holding
them is an extracted Craig zip and is read like the zip. Config keys and
music patterns match accounts after `sanitize`, applied to both sides.

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

TRACK_NUMBER = re.compile(r"^(\d+)-")
# A Craig download without FLAC tracks (AAC, Ogg)
NO_FLAC = (
    "no FLAC tracks; download the recording from Craig as FLAC or as an "
    "Audacity project"
)
# Craig's other download formats and common audio files
_OTHER_AUDIO = {".aac", ".ogg", ".opus", ".m4a", ".mp3", ".wav"}
# A track in a Craig zip: in the *_data folder (Audacity, Audition
# downloads) or at the top (FLAC download)
ZIP_TRACK = re.compile(r"^(?:[^/]+_data/)?[^/]+\.flac$", re.IGNORECASE)


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
        True if any pattern occurs in the account name, ignoring case; both
        are compared after `sanitize`.
    """
    low = sanitize(account).lower()
    return any(sanitize(p).lower() in low for p in patterns)


def speaker_name(speakers: dict[str, str], account: str) -> str:
    """The speaker name for `account`, or the account itself if not listed.

    Args:
        speakers: Sanitized account -> speaker name (``Config.speakers``).
        account: Account as in the track file name.
    """
    return speakers.get(sanitize(account), account)


def account_of(file: Path) -> str:
    """Account of a track file: the stem without a leading ``<n>-``."""
    return TRACK_NUMBER.sub("", file.stem, count=1)


def dir_tracks(folder: Path) -> list[Path]:
    """The ``*.flac`` files directly in `folder`.

    Ordered by the leading track number (``10-x`` after ``2-y``); files
    without a number come last, by name.
    """

    def key(f: Path) -> tuple[int, int, str]:
        m = TRACK_NUMBER.match(f.name)
        return (0, int(m.group(1)), f.name) if m else (1, 0, f.name)

    flacs = [f for f in folder.iterdir() if f.suffix.lower() == ".flac"]
    return sorted((f for f in flacs if f.is_file()), key=key)


def track_dir(folder: Path) -> Path:
    """The folder holding a track folder's FLACs.

    That is `folder` itself, or, when it has no FLACs, its only ``*_data``
    folder with FLACs (a Craig zip extracted as is).
    """
    if dir_tracks(folder):
        return folder
    data = [p for p in folder.glob("*_data") if p.is_dir() and dir_tracks(p)]
    return data[0] if len(data) == 1 else folder


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
        account: Account as in the file name (sanitized by Craig; a track
            from a folder may hold any characters, e.g. ``Game Master``).
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
        info_txt: Text of info.txt; None for a folder without one.
    """

    source: str
    start: datetime
    tracks: list[Track]
    info_txt: str | None = None

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
def open_recording(
    path: Path, music_patterns: list[str], start: datetime | None = None
) -> Iterator[Recording]:
    """Open a Craig zip or a folder of tracks.

    Args:
        path: The Craig zip, or a folder of FLAC tracks (info.txt optional).
        music_patterns: Substrings that mark music bot tracks.
        start: Session start; wins over info.txt. Required for a folder
            without info.txt.

    Yields:
        The recording. Track paths stay valid until the context ends.

    Raises:
        ScribeError: If the input is not a Craig recording, has no FLAC
            files or no start time, or (folder only) a player track listed
            in info.txt has no file.
    """
    path = Path(path)
    if path.is_dir():
        info = path / "info.txt"
        tracks = track_dir(path)
        if info.is_file() and not dir_tracks(tracks) and _other_audio(path):
            # an unpacked Craig download in another format (AAC, Ogg)
            raise ScribeError(f"{path.resolve().name}: {NO_FLAC}")
        yield _read(
            info if info.is_file() else None,
            tracks,
            path.resolve().name,
            music_patterns,
            start,
            # an extracted zip is Craig's own content, checked like the zip
            complete=tracks == path,
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
            root / "info.txt", data_dir, path.name, music_patterns, start, False
        )


def _other_audio(folder: Path) -> bool:
    """Whether `folder` holds audio files that are not FLAC."""
    return any(f.suffix.lower() in _OTHER_AUDIO for f in folder.iterdir())


def _extract(zip_path: Path, dest: Path) -> Path:
    """Extract info.txt and the FLAC tracks; return the folder holding them.

    The project file (.aup, .sesx) and raw.dat (raw data of every track) are
    skipped.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n == "info.txt" or ZIP_TRACK.match(n)]
        if "info.txt" not in names:
            raise ScribeError(f"{zip_path.name}: not a Craig recording (no info.txt)")
        for name in names:
            zf.extract(name, dest)
    tracks = track_dir(dest)
    if not dir_tracks(tracks):
        # e.g. Craig's AAC or Ogg download, or two *_data folders
        raise ScribeError(f"{zip_path.name}: {NO_FLAC}")
    return tracks


def _read(
    info_path: Path | None,
    track_dir: Path,
    source: str,
    music_patterns: list[str],
    start: datetime | None,
    complete: bool,
) -> Recording:
    """Build a Recording from info.txt (if any) and the FLACs in `track_dir`.

    Args:
        info_path: The info.txt file; None for a folder without one.
        track_dir: Folder holding the FLAC tracks.
        source: Name shown in outputs (zip or folder name).
        music_patterns: Substrings that mark music bot tracks.
        start: Given session start; wins over info.txt.
        complete: Require a file for every player listed in info.txt (folder
            input; a missing music track is fine).
    """
    files = dir_tracks(track_dir)
    if not files:
        raise ScribeError(f"{source}: no FLAC files")
    tracks = []
    for f in files:
        account = account_of(f)
        kind = "music" if is_music(account, music_patterns) else "player"
        tracks.append(Track(account, kind, f))
    text = info_path.read_text(encoding="utf-8-sig") if info_path else None
    info = parse_info(text) if text is not None else None
    if start is None:
        if info is None:
            raise ScribeError(
                f"{source}: no info.txt; give the session start with "
                '--start "YYYY-MM-DD HH:MM"'
            )
        start = info.start
    if info is not None and complete:
        present = {sanitize(t.account) for t in tracks}
        missing = [
            a
            for a in info.accounts
            if a not in present and not is_music(a, music_patterns)
        ]
        if missing:
            # also hit by tracks renamed after the speaker next to info.txt
            raise ScribeError(
                f"{source}: info.txt lists {', '.join(missing)}, but no track "
                "file matches; keep Craig's track names, or remove info.txt "
                "and give --start"
            )
    if not any(t.kind == "player" for t in tracks):
        raise ScribeError(f"{source}: no player tracks found")
    return Recording(source, start, tracks, text)
