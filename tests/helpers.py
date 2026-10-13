"""Builders for synthetic test data. Fake accounts and names only."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import numpy as np

from scribe_rpg.transcribe.session import RawSession, Segment, TrackInfo, Word

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


def seg(
    account, start, text, *, word_len=0.4, voiced=True, level_db=-20.0, estimated=False
) -> Segment:
    """Build a segment whose words follow each other from `start`.

    Each word is `word_len` seconds long. `voiced` is one bool for all words or
    a list with one bool per word.
    """
    tokens = text.split()
    flags = voiced if isinstance(voiced, list) else [voiced] * len(tokens)
    words = [
        Word(
            t,
            round(start + i * word_len, 3),
            round(start + (i + 1) * word_len, 3),
            0.9,
            flags[i],
            estimated,
        )
        for i, t in enumerate(tokens)
    ]
    end = words[-1].end
    return Segment(
        account, start, end, text, words, avg_logprob=-0.3, level_db=level_db
    )


def raw_session(segments, tracks=None, **overrides) -> RawSession:
    """Build a raw session with two players and a music track (fake names)."""
    tracks = tracks or [
        TrackInfo("gm_account", "player", "1-gm_account.flac", True),
        TrackInfo("player_one", "player", "2-player_one.flac", True),
        TrackInfo("Kenku_FM_1234", "music", "3-Kenku_FM_1234.flac", False),
    ]
    values = dict(
        date="2026-01-01",
        start="2026-01-01T19:00:00+01:00",
        duration=3600.0,
        source="session.zip",
        whisper={
            "model": "large-v3",
            "initial_prompt": "Vorgrad, Ostromir",
            "prompt_cut": [],
        },
        scribe="0.1.0",
        versions={"whisperx": "3.8.6", "torch": "2.8.0+cu128", "ffmpeg": "7.1"},
    )
    values.update(overrides)
    return RawSession(tracks=tracks, segments=segments, **values)


def tone(seconds: float, amplitude: float = 0.1, sr: int = 16000) -> np.ndarray:
    """440 Hz sine at 16 kHz; amplitude 0.1 is about -23 dBFS RMS."""
    t = np.arange(int(seconds * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def silence(seconds: float, sr: int = 16000) -> np.ndarray:
    """Digital silence at 16 kHz, like a Craig track when nobody talks."""
    return np.zeros(int(seconds * sr), dtype=np.float32)


def make_flac(
    path: Path,
    seconds: float,
    *,
    freq: int = 440,
    volume_db: float = -20,
    on: list[tuple[float, float]] | None = None,
    channels: int = 1,
) -> Path:
    """Write a 48 kHz FLAC sine with ffmpeg (needs ffmpeg in PATH).

    `volume_db` is the tone's peak level in dBFS. `on` lists (start, end)
    seconds where the tone sounds; digital silence elsewhere. None means the
    tone sounds the whole time.
    """
    # lavfi's sine has amplitude 1/8; scale it to the requested peak
    gain = 8 * 10 ** (volume_db / 20)
    level = str(gain)
    if on is not None:
        terms = "+".join(f"between(t,{a},{b})" for a, b in on) or "0"
        level = f"'if({terms},{gain},0)'"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency={freq}:duration={seconds}:sample_rate=48000",
            "-af", f"volume={level}:eval=frame", "-ac", str(channels),
            "-c:a", "flac", str(path),
        ],
        check=True,
    )  # fmt: skip
    return path
