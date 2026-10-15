"""Measure loudness under each word and segment on the speaker's own track.

Craig tracks are digital silence (about -120 dBFS) whenever a player is not
transmitting. A word Whisper "heard" where the track is silent is a
hallucination, so each word gets a ``voiced`` flag here; the cleanup step
later drops segments that are mostly unvoiced.

The measurement runs on the same 16 kHz mono array Whisper used.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from scribe_rpg.transcribe.session import Segment

SAMPLE_RATE = 16_000
SILENCE_DB = -120.0
_BLOCK_SAMPLES = 1 << 18  # samples per float64 block in frame_levels (2 MB)

Audio = npt.NDArray[np.floating[Any]]
Levels = npt.NDArray[np.float64]


def frame_levels(audio: Audio, frame_ms: int, sample_rate: int = SAMPLE_RATE) -> Levels:
    """Compute the RMS level of each frame in dBFS.

    Args:
        audio: Mono samples in the range -1..1.
        frame_ms: Frame length in milliseconds.
        sample_rate: Samples per second of `audio`.

    Returns:
        One level per frame, never below -120 dBFS. The last, partial frame
        is padded with silence.
    """
    n = max(1, int(sample_rate * frame_ms / 1000))
    samples = np.asarray(audio)
    whole = len(samples) // n
    # A view of the whole frames as rows of n samples (no copy of the track).
    frames = samples[: whole * n].reshape(whole, n)
    mean_square = np.empty(whole + (len(samples) % n > 0))
    # float64 per block of rows, so the extra memory stays a few MB.
    rows = max(1, _BLOCK_SAMPLES // n)
    for i in range(0, whole, rows):
        block = frames[i : i + rows].astype(np.float64)
        mean_square[i : i + len(block)] = np.mean(block**2, axis=1)
    if len(mean_square) > whole:
        # The last, partial frame counts as if padded with silence.
        tail = samples[whole * n :].astype(np.float64)
        mean_square[-1] = np.sum(tail**2) / n
    rms = np.sqrt(mean_square)
    # The tiny floor avoids log10(0) on digital silence.
    levels: Levels = np.maximum(20 * np.log10(np.maximum(rms, 1e-12)), SILENCE_DB)
    return levels


def annotate(
    segments: list[Segment],
    audio: Audio,
    energy_db: float,
    frame_ms: int,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Set ``word.voiced`` and ``segment.level_db`` in place.

    A word is voiced if any frame it overlaps is louder than `energy_db`.
    A segment's level is the mean dBFS of its frames louder than
    `energy_db`, or None if there are none.

    Args:
        segments: Segments of one track.
        audio: That track's 16 kHz mono samples.
        energy_db: Threshold in dBFS (``transcribe.cleanup.energy_db``).
        frame_ms: Frame length in milliseconds.
        sample_rate: Samples per second of `audio`.
    """
    levels = frame_levels(audio, frame_ms, sample_rate)
    sec = frame_ms / 1000
    for s in segments:
        for w in s.words:
            span = _span(levels, w.start, w.end, sec)
            w.voiced = bool(span.size) and float(np.max(span)) > energy_db
        span = _span(levels, s.start, s.end, sec)
        loud = span[span > energy_db]
        s.level_db = round(float(loud.mean()), 1) if loud.size else None


def _span(levels: Levels, start: float, end: float, sec: float) -> Levels:
    """Levels of the frames overlapping start..end (at least one frame).

    Frame i covers i*sec .. (i+1)*sec, so the overlapping frames run from
    floor(start/sec) up to, not including, ceil(end/sec).
    """
    # round() first: 0.6 / 0.05 is 11.999999999999998 in floating point.
    first = max(math.floor(round(start / sec, 6)), 0)
    last = max(math.ceil(round(end / sec, 6)), first + 1)
    return levels[first:last]
