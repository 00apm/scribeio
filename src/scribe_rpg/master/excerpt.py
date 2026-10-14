"""Choose the preview excerpt from per-second voice and music activity.

A window of `length` seconds is scored by its least active voice track, so
the preview holds every player. When no window has all voices, a montage
takes each voice's most active stretch instead. `activity` streams a track
through ffmpeg at a low rate; everything else is pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from scribe_rpg import ffmpeg
from scribe_rpg.errors import ScribeError
from scribe_rpg.transcribe.energy import Levels, frame_levels

Activity = npt.NDArray[np.float64]

# Speech activity needs no high frequencies; 8 kHz keeps decoding cheap.
ACTIVITY_RATE = 8000

# Window sums are rounded before comparing, so cumsum rounding noise cannot
# break a tie that is exact in the data.
_DECIMALS = 6


@dataclass(frozen=True)
class Excerpt:
    """The chosen excerpt.

    Attributes:
        ranges: Seconds ``[start, end)``, in time order, not overlapping.
        montage: True if the ranges are a montage of per-voice pieces.
        missing: Voice accounts silent in every range (montage only).
    """

    ranges: list[tuple[int, int]]
    montage: bool
    missing: list[str]


def per_second(levels: Levels, frame_ms: int, energy_db: float) -> Activity:
    """Share of active frames (level above `energy_db`) in each second.

    The last, partial second is kept and averaged over the frames it has.
    """
    active = (np.asarray(levels) > energy_db).astype(np.float64)
    if not len(active):
        return np.zeros(0)
    per_sec = max(1, round(1000 / frame_ms))
    starts = np.arange(0, len(active), per_sec)
    counts = np.diff(np.append(starts, len(active)))
    shares: Activity = np.add.reduceat(active, starts) / counts
    return shares


def activity(path: Path, frame_ms: int, energy_db: float) -> Activity:
    """Activity per second of one track, decoded in one-minute chunks.

    A minute holds whole frames for any frame length that divides 60 s, so
    only the last chunk's last frame is padded.
    """
    levels = [
        frame_levels(chunk, frame_ms, ACTIVITY_RATE)
        for chunk in ffmpeg.decode(path, ACTIVITY_RATE, chunk_seconds=60)
    ]
    joined = np.concatenate(levels) if levels else np.zeros(0)
    return per_second(joined, frame_ms, energy_db)


def activities(paths: list[Path], frame_ms: int, energy_db: float) -> list[Activity]:
    """`activity` of every track, in parallel (one ffmpeg per CPU), in order."""
    return ffmpeg.parallel(lambda p: activity(p, frame_ms, energy_db), paths)


def choose(
    voice: dict[str, Activity],
    music: list[Activity],
    length: int,
    at: int | None = None,
    montage: bool = False,
) -> Excerpt:
    """Choose the excerpt: the best window, a montage, or the one at `at`.

    Args:
        voice: Account -> activity of each voice (non-music) track.
        music: Activity of each music track (breaks ties).
        length: Excerpt length in seconds; clipped to the recording.
        at: Start second given by the user; skips the search.
        montage: Force a montage.

    Raises:
        ScribeError: If every track is empty, or `at` is at or past the end
            of the recording.
    """
    duration = max((len(a) for a in [*voice.values(), *music]), default=0)
    if duration == 0:
        raise ScribeError("no audio in the recording (every track is empty)")
    if at is not None:
        if at >= duration:
            raise ScribeError(f"--at {hms(at)} is past the end ({hms(duration)})")
        return Excerpt([(at, min(at + length, duration))], False, [])
    length = max(1, min(length, duration))
    names = list(voice)
    voices = _pad([voice[n] for n in names], duration)
    if not montage and len(names):
        start = _best_window(voices, _pad(music, duration).sum(axis=0), length)
        if start is not None:
            return Excerpt([(start, start + length)], False, [])
    if not len(names):
        return Excerpt([(0, length)], False, [])
    return _montage(names, voices, length)


def hms(seconds: int) -> str:
    """``3725`` -> ``"01:02:05"``."""
    h, rest = divmod(int(seconds), 3600)
    return f"{h:02d}:{rest // 60:02d}:{rest % 60:02d}"


def _pad(arrays: list[Activity], duration: int) -> npt.NDArray[np.float64]:
    """Stack the arrays as rows, zero-padded to `duration` seconds."""
    out = np.zeros((len(arrays), duration))
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return out


def _window_sums(rows: npt.NDArray[np.float64], length: int) -> npt.NDArray[np.float64]:
    """Sum of each row over every window of `length` (1 s steps), rounded."""
    c = np.cumsum(rows, axis=-1)
    c = np.concatenate([np.zeros(c.shape[:-1] + (1,)), c], axis=-1)
    sums: npt.NDArray[np.float64] = np.round(
        c[..., length:] - c[..., :-length], _DECIMALS
    )
    return sums


def _best_window(
    voices: npt.NDArray[np.float64], music: npt.NDArray[np.float64], length: int
) -> int | None:
    """Start of the best window; None if every window misses a voice.

    Best = highest least-active-voice sum, then most music, then earliest.
    """
    score = _window_sums(voices, length).min(axis=0)
    music_sum = _window_sums(music, length)
    # lexsort sorts by the last key first; the index keeps ties earliest.
    order = np.lexsort((np.arange(len(score)), -music_sum, -score))
    best = int(order[0])
    return best if score[best] > 0 else None


def _montage(names: list[str], voices: npt.NDArray[np.float64], length: int) -> Excerpt:
    """Each voice's most active piece of ``length // voices`` s, merged."""
    piece = max(1, length // len(names))
    sums = _window_sums(voices, piece)
    # argmax returns the first maximum: earliest on ties.
    starts = sorted(int(s) for s in sums.argmax(axis=1))
    ranges: list[tuple[int, int]] = []
    for s in starts:
        if ranges and s <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], s + piece))
        else:
            ranges.append((s, s + piece))
    missing = [
        name
        for name, row in zip(names, voices, strict=True)
        if not any(row[s:e].sum() > 0 for s, e in ranges)
    ]
    return Excerpt(ranges, True, missing)
