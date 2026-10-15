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
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
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
    parse_integrated,
    parse_peak,
    passes,
    percent_of,
    track_gains,
)

OnMeasured = Callable[[str, list[Measured]], None]
# Told a row (account or "mix"), a pass's phase ("level 1/2") and percent done
OnRowPass = Callable[[str, str, float], None]

# ffmpeg's own name for a bad option ("... out of range [1 - 20]"), stripped
# of its "[Parsed_x @ 0x...] " prefix
_PREFIX = re.compile(r"^\[[^\]]*\]\s*")
# Silence after each track before amix: ffmpeg 6.1's amix drops the last
# ~1.5 s of an input at its end (7.1 does not). The pad is what gets lost;
# the render cuts the mix to the longest track (-t), and loudness gating
# ignores the silence in the measuring runs.
MIX_PAD = "apad=pad_dur=5"


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
    split: bool = False,
) -> str:
    """The whole filter graph, ending in the ``[out]`` label.

    With `split`, each track's processed audio is also labelled ``[k<i>]``
    (``asplit`` after its chain, before any copy to stereo for the mix).
    """
    mix_chain = chain_filter(mix_steps, mix_gains)
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
        parts.append(f"{track}{to_mix(p, channels)},{MIX_PAD}[t{i}]")
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
    on_pass: OnRowPass | None = None,
) -> Gains:
    """Measure every loudness step: tracks in parallel, then the mix.

    Fills ``plan.gains`` of every plan and returns the mix gains.
    `on_measured` gets each track's measurements as soon as that track is
    done (from its worker thread), then the mix's. `on_pass` is told, per
    row (the account, or ``mix``), each pass's phase and percent done.
    """

    def one(plan: TrackPlan) -> None:
        account = plan.track.account
        row_pass = partial(on_pass, account) if on_pass else None
        prepared = prep(plan, channels)
        plan.gains, measured = track_gains(
            plan.track.path, prepared, plan.steps[: plan.upto], account, row_pass
        )
        if on_measured:
            on_measured(account, measured)

    ffmpeg.parallel(one, plans)
    mix_gains: Gains = {}
    mix_measured = []
    found = passes(mix_steps)
    length = max(ffmpeg.duration(p.track.path) for p in plans) if found else 0.0
    for i, phase in found:
        step = mix_steps[i]
        on_time = (
            percent_of(partial(on_pass, "mix"), phase, length) if on_pass else None
        )
        text = _graph(plans, channels, measure_filter(mix_steps, mix_gains, i))
        level = measure(_inputs(plans), text, f"measure mix, {step.name}", on_time)
        gain = None if level is None else float(step.options["target"]) - level
        mix_gains[i] = gain
        mix_measured.append(Measured(step.name, level, gain))
    if on_measured and mix_measured:
        on_measured("mix", mix_measured)
    return mix_gains


def mix_level(
    plans: list[TrackPlan],
    mix_steps: list[Step],
    mix_gains: Gains,
    channels: str,
    label: str,
) -> tuple[float | None, float | None]:
    """Loudness and sample peak of the mix, in one run (output discarded).

    Returns:
        (integrated loudness in LUFS, sample peak in dBFS); each None if
        the mix is silent.
    """
    meter = "ebur128=peak=sample:dualmono=true:framelog=verbose"
    text = _graph(plans, channels, f"{chain_filter(mix_steps, mix_gains)},{meter}")
    args = [*_inputs(plans), "-filter_complex", text, "-f", "null", "-"]
    stderr = ffmpeg.run(args, f"measure {label}").stderr
    return parse_integrated(stderr), parse_peak(stderr)


def partial_path(path: Path) -> Path:
    """Where `path` is written until it is complete: ``s1_master.partial.mp3``,
    ``s1_tracks.partial``. A leftover of a hard stop names itself."""
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def render(
    plans: list[TrackPlan],
    mix_steps: list[Step],
    mix_gains: Gains,
    channels: str,
    out: Path,
    fmt: str,
    bitrate: str,
    on_progress: Callable[[float], None] | None = None,
    tracks_dir: Path | None = None,
) -> None:
    """The final run: all tracks, chains, mix, mix chain, encode, to `out`.

    `on_progress` gets percent done (0..100). With `tracks_dir`, the same run
    also writes each processed track as FLAC (32-bit), under its input file
    name, into ``partial_path(tracks_dir)`` (an existing folder; it may already
    hold e.g. info.txt). Outputs are written under `partial_path` names and
    replace `out` and `tracks_dir` only once the run succeeds; if it fails or
    is interrupted, the partial outputs are removed and old ones are kept.
    """
    part_out = partial_path(out)
    part_dir = partial_path(tracks_dir) if tracks_dir is not None else None
    try:
        # every output is cut to its input's length: filters such as the eq
        # leave a short ring-out tail after the last sample
        lengths = [ffmpeg.duration(p.track.path) for p in plans]
        total = max(lengths)
        on_time = None
        if on_progress is not None:
            report = on_progress

            def on_time(seconds: float) -> None:
                report(min(100.0, 100 * seconds / (total or 1.0)))

        split = part_dir is not None
        args = [
            *_inputs(plans),
            "-filter_complex",
            graph(plans, mix_steps, mix_gains, channels, split),
            "-map", "[out]", *encode_args(fmt, bitrate), "-t", f"{total:.6f}",
            "-y", str(part_out),
        ]  # fmt: skip
        if part_dir is not None:
            for i, (p, length) in enumerate(zip(plans, lengths, strict=True)):
                args += ["-map", f"[k{i}]", "-c:a", "flac", "-sample_fmt", "s32",
                         "-t", f"{length:.6f}", "-y",
                         str(part_dir / p.track.path.name)]  # fmt: skip
        ffmpeg.run_progress(args, f"render {out.name}", on_time)
    except BaseException:
        discard(part_out)
        if part_dir is not None:
            discard(part_dir)
        raise
    os.replace(part_out, out)
    if tracks_dir is not None and part_dir is not None:
        if tracks_dir.exists():
            shutil.rmtree(tracks_dir)
        part_dir.rename(tracks_dir)


def discard(path: Path, wait: float = 2.0) -> None:
    """Remove a file or folder if possible; never raises.

    A killed ffmpeg started through a shim (Chocolatey, Scoop) may hold its
    output files for a moment, so removal is retried for up to `wait` seconds.
    """
    deadline = time.monotonic() + wait
    while path.exists():
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError:
            if time.monotonic() >= deadline:
                return
            time.sleep(0.1)


def check(cfg: MasterConfig) -> None:
    """Run every filter step once on 0.1 s of silence, before any real work.

    ffmpeg checks option names and ranges itself; a bad one stops here, with
    the step's config path, instead of after minutes of measuring.

    The encoder is tried too (format and bitrate), so a bitrate it refuses
    or an ffmpeg build without it stops here as well.

    Raises:
        ScribeError: ``<path> (<type>): <ffmpeg's message>``, first bad step;
            then ``master.format`` / ``master.bitrate`` for the encoder.
    """
    chains = [cfg.voice, cfg.music, cfg.mix, *cfg.players.values()]
    steps = {s.where: s for chain in chains for s in chain if s.type not in OPERATIONS}
    problems = ffmpeg.parallel(_problem, list(steps.values()))
    for step, problem in zip(steps.values(), problems, strict=True):
        if problem is not None:
            raise ScribeError(f"{step.where} ({step.type}): {problem}")
    _check_encoder(cfg.format, cfg.bitrate)


def _silence(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg on 0.1 s of mono silence with `args` before the null output."""
    return ffmpeg.attempt(
        ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "0.1",
         *args, "-f", "null", "-"]
    )  # fmt: skip


def _problem(step: Step) -> str | None:
    """ffmpeg's complaint about `step`, or None if it runs."""
    # chain_filter, as the render builds it (the eq's delay cut included)
    done = _silence(["-af", chain_filter([step], {})])
    if done.returncode == 0:
        return None
    lines = _lines(done.stderr)
    # the most specific line first: a range, then any option line, else the last
    for key in ("out of range", "option"):
        for line in lines:
            if key in line.lower():
                return line
    return lines[-1] if lines else f"ffmpeg exited with {done.returncode}"


def _check_encoder(fmt: str, bitrate: str) -> None:
    """Encode the silence as the master will be encoded.

    Raises:
        ScribeError: If the encoder is missing or refuses the bitrate.
    """
    args = encode_args(fmt, bitrate)
    # errors only: the stream info lines would come before the reason
    done = _silence(["-loglevel", "error", *args])
    if done.returncode == 0:
        return
    codec = args[1]  # "-c:a", codec, ...
    if f"Unknown encoder '{codec}'" in done.stderr:
        raise ScribeError(
            f"master.format ({fmt}): this ffmpeg has no {codec} encoder; install "
            f"an ffmpeg build with {codec}, or choose another master.format "
            "(or --format)"
        )
    lines = _lines(done.stderr)
    # the encoder's own complaint comes first (e.g. "The bit rate ... is
    # unsupported"); the lines after it are ffmpeg's generic follow-ups
    first = lines[0] if lines else f"ffmpeg exited with {done.returncode}"
    raise ScribeError(f"master.bitrate ({fmt} {bitrate}): {first}")


def _lines(stderr: str) -> list[str]:
    """ffmpeg's non-empty error lines without their ``[name @ 0x...]`` prefix."""
    lines = [_PREFIX.sub("", line).strip() for line in stderr.splitlines()]
    return [line for line in lines if line]
