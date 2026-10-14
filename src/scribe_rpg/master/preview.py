"""Preview files in ``<name>_preview/``: cut tracks and listening files.

The preview folder is scratch: replaced on every ``--preview`` run.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from scribe_rpg import craig, ffmpeg
from scribe_rpg.master import mix
from scribe_rpg.master.catalogue import Step
from scribe_rpg.master.config import MasterConfig
from scribe_rpg.progress import Progress

FADE = 0.02  # seconds of fade at each montage join
# Preview files are matched to one loudness, so they are compared by tone and
# dynamics, not volume: at most MATCH_TARGET, lower if a file's sample peak
# would otherwise go above MATCH_PEAK (FLAC is integer: a peak above 0 clips).
MATCH_TARGET = -19.0  # LUFS
MATCH_PEAK = -1.0  # dBFS


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


@dataclass(frozen=True)
class PreviewFile:
    """One listening file: the excerpt mixed, every chain run up to a step.

    Attributes:
        name: File name, e.g. ``03-voice-comp.flac`` or ``08-final.mp3``.
        plans: Track plans, each with its ``upto``.
        mix_steps: The prefix of the mix chain this file uses.
        fmt: ``flac``, or the master's format for the final file.
        channels: ``mono`` or ``stereo``.
    """

    name: str
    plans: list[mix.TrackPlan]
    mix_steps: list[Step]
    fmt: str
    channels: str


def _same_chain(steps: list[Step], voice: list[Step]) -> bool:
    return [s.name for s in steps] == [s.name for s in voice]


def step_files(
    full: list[mix.TrackPlan], cfg: MasterConfig, before_after: bool = False
) -> list[PreviewFile]:
    """The listening files, in order (pure: nothing runs).

    File k is cumulative: every voice chain up to its step k, never one
    filter on raw audio. Numbers follow the config's chains, so a name is
    stable: voice (music) files are skipped when no track is a voice
    (music), but their numbers are not reused. Voice files leave the music
    out: it is still untreated there and would cover the voices; it joins
    at its first music step. A voice with its own chain (other step names
    than ``voice``) is fully processed in every voice file.

    Args:
        full: One plan per track, with its full chain and measured gains.
        cfg: The resolved master config.
        before_after: Only ``00-original`` and the final file.
    """

    def file(
        nn: int, name: str, upto: Callable[[mix.TrackPlan], int | None],
        mix_steps: list[Step], fmt: str = "flac",
        among: list[mix.TrackPlan] = full,
    ) -> PreviewFile:  # fmt: skip
        plans = [replace(p, upto=upto(p)) for p in among]
        return PreviewFile(f"{nn:02d}-{name}", plans, mix_steps, fmt, cfg.channels)

    def is_music(p: mix.TrackPlan) -> bool:
        return p.track.kind == "music"

    nv, nm, nx = len(cfg.voice), len(cfg.music), len(cfg.mix)
    original = PreviewFile(
        "00-original.flac", [replace(p, steps=[], upto=None) for p in full], [],
        "flac", cfg.channels,
    )  # fmt: skip
    final = file(1 + nv + nm + nx, f"final.{cfg.format}", lambda p: None,
                 cfg.mix, cfg.format)  # fmt: skip
    if before_after:
        return [original, final]
    files = [original]
    voices = [p for p in full if not is_music(p)]
    for k, step in enumerate(cfg.voice if voices else []):

        def voice_upto(p: mix.TrackPlan, k: int = k) -> int | None:
            return k + 1 if _same_chain(p.steps, cfg.voice) else None

        name = f"voice-{step.name}.flac"
        files.append(file(k + 1, name, voice_upto, [], among=voices))
    if any(is_music(p) for p in full):
        for j, step in enumerate(cfg.music):

            def music_upto(p: mix.TrackPlan, j: int = j) -> int | None:
                return j + 1 if is_music(p) else None

            files.append(file(nv + j + 1, f"music-{step.name}.flac", music_upto, []))
    for m, step in enumerate(cfg.mix):
        files.append(
            file(
                nv + nm + m + 1,
                f"mix-{step.name}.flac",
                lambda p: None,
                cfg.mix[: m + 1],
            )  # fmt: skip
        )
    return [*files, final]


def own_chains(full: list[mix.TrackPlan], cfg: MasterConfig) -> list[str]:
    """Accounts of voice tracks whose chain differs from ``voice`` (by names)."""
    return [
        p.track.account
        for p in full
        if p.track.kind != "music" and not _same_chain(p.steps, cfg.voice)
    ]


def common_level(levels: list[tuple[float | None, float | None]]) -> float:
    """The one loudness every preview file is matched to.

    At most MATCH_TARGET, and low enough that no file's sample peak goes
    above MATCH_PEAK once matched. Silent files are ignored.

    Args:
        levels: (integrated LUFS, sample peak dBFS) of each file, unmatched.
    """
    level = MATCH_TARGET
    for loudness, peak in levels:
        if loudness is not None and peak is not None:
            level = min(level, MATCH_PEAK - peak + loudness)
    return level


def write_files(
    tracks: list[craig.Track],
    cfg: MasterConfig,
    dest: Path,
    progress: Progress,
    on_measured: mix.OnMeasured | None = None,
    before_after: bool = False,
) -> list[Path]:
    """Write the listening files for the cut `tracks` into `dest`.

    Every loudness step is measured once, on the full chains; each file
    reuses those gains for its prefix. Then each file's loudness and peak
    are measured, one common level is chosen (`common_level`) and every file
    is written matched to it. Files are measured and written in parallel.

    Returns:
        The written paths, in file order.
    """
    full = [
        mix.TrackPlan(t, cfg.chain_for(t), stereo=cfg.keeps_stereo(t)) for t in tracks
    ]
    with progress.step(f"measure {len(full)} tracks"):
        mix_gains = mix.measure_all(full, cfg.mix, cfg.channels, on_measured)
    files = step_files(full, cfg, before_after)
    if not before_after:
        for account in own_chains(full, cfg):
            progress.console.print(
                f"{account}: own chain, fully processed from file 01", markup=False
            )

    def level_of(f: PreviewFile) -> tuple[float | None, float | None]:
        return mix.mix_level(f.plans, f.mix_steps, mix_gains, f.channels, f.name)

    with progress.step(f"measure {len(files)} files"), _pool() as pool:
        levels = list(pool.map(level_of, files))
    level = common_level(levels)
    lowered = " (lowered: the loudest peak stays at -1 dBFS)"
    note = lowered if level < MATCH_TARGET else ""
    progress.console.print(f"preview level: {level:.1f} LUFS{note}", markup=False)

    def write(item: tuple[PreviewFile, float | None]) -> Path:
        f, loudness = item
        match = Step("match", "loudness", {"target": level}, "preview")
        gain = None if loudness is None else level - loudness
        gains = {**mix_gains, len(f.mix_steps): gain}
        out = dest / f.name
        mix.render(f.plans, [*f.mix_steps, match], gains, f.channels, out,
                   f.fmt, cfg.bitrate)  # fmt: skip
        return out

    loudness = [lv for lv, _ in levels]
    with progress.step(f"write {len(files)} files"), _pool() as pool:
        return list(pool.map(write, zip(files, loudness, strict=True)))


def _pool() -> ThreadPoolExecutor:
    # one ffmpeg graph uses about one core
    return ThreadPoolExecutor(max_workers=os.cpu_count() or 1)
