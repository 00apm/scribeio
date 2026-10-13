"""Preview files: the excerpt cut out of every track, in ``<name>_preview/``.

The preview folder is scratch: replaced on every ``--preview`` run.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scribe_rpg import craig, ffmpeg

FADE = 0.02  # seconds of fade at each montage join


def fresh_folder(path: Path) -> Path:
    """Empty `path` (a ``*_preview`` folder), creating it if needed.

    Raises:
        ValueError: If the folder name does not end with ``_preview`` (a
            guard: this deletes everything inside).
    """
    if not path.name.endswith("_preview"):
        raise ValueError(f"not a preview folder: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def cut(
    tracks: list[craig.Track], ranges: list[tuple[int, int]], dest: Path
) -> list[Path]:
    """Cut `ranges` out of every track into `dest`, as FLAC, same file names.

    Every track gets the same ranges, so the cut tracks stay aligned. Several
    ranges are joined in order with short fades at each join. One ffmpeg run
    per track, in parallel.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        return list(pool.map(lambda t: _cut_one(t, ranges, dest), tracks))


def _cut_one(track: craig.Track, ranges: list[tuple[int, int]], dest: Path) -> Path:
    out = dest / track.path.name
    if len(ranges) == 1:
        [(start, end)] = ranges
        args = ["-ss", str(start), "-t", str(end - start), "-i", str(track.path)]
    else:
        args = ["-i", str(track.path), "-filter_complex", _join(ranges),
                "-map", "[out]"]  # fmt: skip
    ffmpeg.run([*args, "-c:a", "flac", "-y", str(out)], f"cut {track.path.name}")
    return out


def _join(ranges: list[tuple[int, int]]) -> str:
    """Filter graph: trim each range, fade the joins, concatenate."""
    pieces = []
    last = len(ranges) - 1
    for i, (start, end) in enumerate(ranges):
        chain = f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS"
        if i > 0:
            chain += f",afade=t=in:d={FADE}"
        if i < last:
            chain += f",afade=t=out:st={end - start - FADE:.3f}:d={FADE}"
        pieces.append(f"{chain}[p{i}]")
    labels = "".join(f"[p{i}]" for i in range(len(ranges)))
    return ";".join(pieces) + f";{labels}concat=n={len(ranges)}:v=0:a=1[out]"
