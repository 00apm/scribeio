"""Preview files in ``<name>_preview/``: cut tracks and listening files.

The preview folder is scratch: replaced on every ``--preview`` run.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scribe_rpg import craig, ffmpeg
from scribe_rpg.master import mix
from scribe_rpg.master.config import MasterConfig
from scribe_rpg.progress import Progress

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


def write_files(
    tracks: list[craig.Track], cfg: MasterConfig, dest: Path, progress: Progress
) -> list[Path]:
    """Write the listening files for the cut `tracks` into `dest`.

    ``00-original.flac`` is the plain mix; ``NN-final.<format>`` is what
    ``scribe master`` makes of the excerpt (NN = 1 + number of steps, so step
    files can sit in between). Both are brought to the same loudness (MATCH).
    """
    total = len(cfg.voice) + len(cfg.music) + len(cfg.mix)

    def track_plans(chained: bool) -> list[mix.TrackPlan]:
        return [
            mix.TrackPlan(
                t, cfg.chain_for(t) if chained else [], stereo=cfg.keeps_stereo(t)
            )
            for t in tracks
        ]

    files = [
        (dest / "00-original.flac", track_plans(False), [], "flac"),
        (
            dest / f"{total + 1:02d}-final.{cfg.format}",
            track_plans(True),
            cfg.mix,
            cfg.format,
        ),
    ]
    written = []
    for out, plans, mix_steps, fmt in files:
        with progress.step(f"write {out.name}"):
            steps = [*mix_steps, mix.MATCH]
            gains = mix.measure_all(plans, steps, cfg.channels)
            mix.render(plans, steps, gains, cfg.channels, out, fmt, cfg.bitrate)
        written.append(out)
    return written
