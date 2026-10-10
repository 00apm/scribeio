"""Builders for synthetic test data. Fake accounts and names only."""

from __future__ import annotations

import zipfile
from pathlib import Path

START = "2026-01-01T18:00:00.000Z"

# (entry in info.txt, account in the track file name)
TRACKS = [
    ("gm.account#0 (000000000000000010)", "gm_account"),
    ("player.one#0 (000000000000000011)", "player_one"),
    ("Kenku FM#1234 (000000000000000012)", "Kenku_FM_1234"),
]

MUSIC = ["kenku", "jukebox"]  # default + one custom bot name


def info_txt(tracks=TRACKS, start=START) -> str:
    """Return an info.txt laid out like Craig's, with fake guild and channel."""
    entries = "\n".join(f"\t{entry}" for entry, _ in tracks)
    return (
        "Recording AAAAAA\n\n"
        "Guild:\t\tTest Guild (000000000000000001)\n"
        "Channel:\ttest-channel (000000000000000002)\n"
        "Requester:\tgm.account#0 (000000000000000010)\n"
        f"Start time:\t{start}\n\n"
        f"Tracks:\n{entries}\n"
    )


def craig_zip(folder: Path, tracks=TRACKS, start=START, name="session.zip") -> Path:
    """Write a zip laid out like Craig's: info.txt, .aup, *_data/*.flac, raw.dat."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("info.txt", info_txt(tracks, start))
        zf.writestr("AAAAAA.aup", "<project/>")
        zf.writestr("raw.dat", b"\0" * 16)
        for n, (_, account) in enumerate(tracks, 1):
            zf.writestr(f"AAAAAA_data/{n}-{account}.flac", b"fLaC fake")
    return path


def craig_dir(folder: Path, tracks=TRACKS, start=START, skip=()) -> Path:
    """Write a folder of processed tracks: info.txt + <n>-<account>.flac.

    Accounts listed in `skip` get no file (to test missing tracks).
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "info.txt").write_text(info_txt(tracks, start), encoding="utf-8")
    for n, (_, account) in enumerate(tracks, 1):
        if account not in skip:
            (folder / f"{n}-{account}.flac").write_bytes(b"fLaC fake")
    return folder
