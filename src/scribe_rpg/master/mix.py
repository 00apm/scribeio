"""The ffmpeg runs of mastering: loudness measurements, then one final run.

Every track goes through its chain, the tracks are summed with ``amix``
(``normalize=0``: no division by the number of inputs; ``duration=longest``),
the mix goes through the mix chain and is encoded, all in one ffmpeg run.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from scribe_rpg import craig, ffmpeg
from scribe_rpg.errors import ScribeError
from scribe_rpg.master.catalogue import OPERATIONS, Step
from scribe_rpg.master.config import MasterConfig
from scribe_rpg.master.render import (
    Gains,
    Measured,
    chain_filter,
    measure,
    measure_filter,
    step_filter,
    track_gains,
)

# Appended to the mix chain of every preview file: all files end at the same
# loudness, so they are compared by tone and dynamics, not volume.
MATCH = Step("match", "loudness", {"target": -19, "dual_mono": True}, "preview")

OnMeasured = Callable[[str, list[Measured]], None]

# ffmpeg's own name for a bad option ("... out of range [1 - 20]"), stripped
# of its "[Parsed_x @ 0x...] " prefix
_PREFIX = re.compile(r"^\[[^\]]*\]\s*")


@dataclass
class TrackPlan:
    """One track with its chain and the gains measured for it.

    Attributes:
        track: The input track.
        steps: Its resolved chain.
        gains: Measured gain of each loudness step (by chain index).
        upto: Use only ``steps[:upto]`` (preview step files); None = all.
        stereo: Keep the track's stereo when the output is stereo (music, a
            voice recorded in stereo); otherwise it is processed as mono.
    """

    track: craig.Track
    steps: list[Step]
    gains: Gains = field(default_factory=dict)
    upto: int | None = None
    stereo: bool = False


def _layout(plan: TrackPlan, channels: str) -> str:
    """Channel layout the track is processed in."""
    return "stereo" if channels == "stereo" and plan.stereo else "mono"


def prep(plan: TrackPlan, channels: str) -> str:
    """Input filter of a track: float, in the layout it is processed in."""
    return f"aformat=sample_fmts=flt:channel_layouts={_layout(plan, channels)}"


def to_mix(plan: TrackPlan, channels: str) -> str:
    """Filter text after the chain: a mono track in a stereo mix is copied to
    both sides at full level (as its loudness was measured, dual mono); empty
    otherwise. Done once, after the chain, so the chain filters run once.
    """
    if channels == "stereo" and _layout(plan, channels) == "mono":
        return ",pan=stereo|c0=c0|c1=c0"
    return ""


def graph(
    plans: list[TrackPlan],
    mix_steps: list[Step],
    mix_gains: Gains,
    channels: str,
    mix_upto: int | None = None,
    split: bool = False,
) -> str:
    """The whole filter graph, ending in the ``[out]`` label.

    With `split`, each track's processed audio is also labelled ``[k<i>]``
    (``asplit`` after its chain, before any copy to stereo for the mix).
    """
    mix_chain = chain_filter(mix_steps, mix_gains, mix_upto)
    return _graph(plans, channels, mix_chain, split) + "[out]"


def _graph(
    plans: list[TrackPlan], channels: str, mix_chain: str, split: bool = False
) -> str:
    """Per-track chains, the amix, then `mix_chain` (no output label)."""
    parts = []
    for i, p in enumerate(plans):
        track = f"[{i}:a]{prep(p, channels)},{chain_filter(p.steps, p.gains, p.upto)}"
        if split:
            track += f",asplit=2[k{i}][m{i}];[m{i}]anull"
        parts.append(f"{track}{to_mix(p, channels)}[t{i}]")
    labels = "".join(f"[t{i}]" for i in range(len(plans)))
    amix = f"{labels}amix=inputs={len(plans)}:normalize=0:duration=longest"
    return ";".join(parts) + f";{amix},{mix_chain}"


def _inputs(plans: list[TrackPlan]) -> list[str]:
    return [arg for p in plans for arg in ("-i", str(p.track.path))]


def encode_args(fmt: str, bitrate: str) -> list[str]:
    """Codec arguments: mp3 (CBR), m4a (aac), opus, or flac (lossless)."""
    codecs = {"mp3": "libmp3lame", "m4a": "aac", "opus": "libopus"}
    if fmt == "flac":
        return ["-c:a", "flac"]
    return ["-c:a", codecs[fmt], "-b:a", bitrate]


def measure_all(
    plans: list[TrackPlan],
    mix_steps: list[Step],
    channels: str,
    on_measured: OnMeasured | None = None,
) -> Gains:
    """Measure every loudness step: tracks in parallel, then the mix.

    Fills ``plan.gains`` of every plan and returns the mix gains.
    `on_measured` gets each track's (and the mix's) measurements, in order.
    """

    def one(plan: TrackPlan) -> list[Measured]:
        prepared = prep(plan, channels)
        plan.gains, measured = track_gains(
            plan.track.path, prepared, plan.steps[: plan.upto], plan.track.account
        )
        return measured

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        results = list(pool.map(one, plans))
    if on_measured:
        for plan, measured in zip(plans, results, strict=True):
            on_measured(plan.track.account, measured)
    mix_gains: Gains = {}
    mix_measured = []
    for i, step in enumerate(mix_steps):
        if step.type != "loudness":
            continue
        text = _graph(plans, channels, measure_filter(mix_steps, mix_gains, i))
        level = measure(_inputs(plans), text, f"measure mix, {step.name}")
        gain = None if level is None else float(step.options["target"]) - level
        mix_gains[i] = gain
        mix_measured.append(Measured(step.name, level, gain))
    if on_measured and mix_measured:
        on_measured("mix", mix_measured)
    return mix_gains


def render(
    plans: list[TrackPlan],
    mix_steps: list[Step],
    mix_gains: Gains,
    channels: str,
    out: Path,
    fmt: str,
    bitrate: str,
    on_progress: Callable[[float], None] | None = None,
    mix_upto: int | None = None,
    tracks_dir: Path | None = None,
) -> None:
    """The final run: all tracks, chains, mix, mix chain, encode, to `out`.

    `on_progress` gets percent done (0..100). With `tracks_dir` (an existing,
    empty folder), the same run also writes each processed track there as
    FLAC (32-bit), under its input file name. If anything fails or the run is
    interrupted, `out` and `tracks_dir` are removed: half-written outputs
    would otherwise block the next run as "already exists".
    """
    # every output is cut to its input's length: filters such as the eq leave
    # a short ring-out tail after the last sample
    lengths = [ffmpeg.duration(p.track.path) for p in plans]
    total = max(lengths)
    on_time = None
    if on_progress is not None:
        report = on_progress

        def on_time(seconds: float) -> None:
            report(min(100.0, 100 * seconds / (total or 1.0)))

    split = tracks_dir is not None
    args = [
        *_inputs(plans),
        "-filter_complex",
        graph(plans, mix_steps, mix_gains, channels, mix_upto, split),
        "-map", "[out]", *encode_args(fmt, bitrate), "-t", f"{total:.6f}",
        "-y", str(out),
    ]  # fmt: skip
    if tracks_dir is not None:
        for i, (p, length) in enumerate(zip(plans, lengths, strict=True)):
            args += ["-map", f"[k{i}]", "-c:a", "flac", "-sample_fmt", "s32",
                     "-t", f"{length:.6f}", "-y",
                     str(tracks_dir / p.track.path.name)]  # fmt: skip
    try:
        ffmpeg.run_progress(args, f"render {out.name}", on_time)
    except BaseException:
        out.unlink(missing_ok=True)
        if tracks_dir is not None:
            shutil.rmtree(tracks_dir, ignore_errors=True)
        raise


def check(cfg: MasterConfig) -> None:
    """Run every filter step once on 0.1 s of silence, before any real work.

    ffmpeg checks option names and ranges itself; a bad one stops here, with
    the step's config path, instead of after minutes of measuring.

    Raises:
        ScribeError: ``<path> (<type>): <ffmpeg's message>``, first bad step.
    """
    chains = [cfg.voice, cfg.music, cfg.mix, *cfg.players.values()]
    steps = {s.where: s for chain in chains for s in chain if s.type not in OPERATIONS}
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        problems = list(pool.map(_problem, steps.values()))
    for step, problem in zip(steps.values(), problems, strict=True):
        if problem is not None:
            raise ScribeError(f"{step.where} ({step.type}): {problem}")


def _problem(step: Step) -> str | None:
    """ffmpeg's complaint about `step`, or None if it runs."""
    done = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-nostats",
         "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "0.1",
         "-af", f"aformat=sample_fmts=flt,{step_filter(step)}",
         "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )  # fmt: skip
    if done.returncode == 0:
        return None
    lines = [_PREFIX.sub("", line).strip() for line in done.stderr.splitlines()]
    lines = [line for line in lines if line]
    # the most specific line first: a range, then any option line, else the last
    for key in ("out of range", "option"):
        for line in lines:
            if key in line.lower():
                return line
    return lines[-1] if lines else f"ffmpeg exited with {done.returncode}"
