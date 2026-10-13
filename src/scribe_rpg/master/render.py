"""Filter text for ffmpeg built from resolved steps, and loudness measurement.

Building filter text is pure; `measure` and `track_gains` run ffmpeg with the
output discarded (no temporary audio files).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from scribe_rpg import ffmpeg
from scribe_rpg.master.catalogue import Step, quote

# Every chain starts in 32-bit float, so no step clips between steps.
FLOAT = "aformat=sample_fmts=flt"
# Follows every filter step: filter tails decaying over digital silence become
# denormals (floats so small the CPU handles them ~100x slower); adenorm adds
# a -351 dB offset, far below 16-bit resolution, so the output is unchanged.
DENORM = "adenorm"
# firequalizer's default delay: its output lags its input by `delay` seconds
EQ_DELAY = 0.01

Gains = dict[int, float | None]  # chain index of a loudness step -> dB (None: silent)

# At or below this, BS.1770 gating found nothing to measure: a silent track.
SILENT_LUFS = -70.0
_INTEGRATED = re.compile(r"^\s*I:\s*(-?inf|-?\d+(?:\.\d+)?)\s*LUFS", re.MULTILINE)


@dataclass(frozen=True)
class Measured:
    """One measured loudness step.

    Attributes:
        step: Step name.
        loudness: Integrated loudness before the step, LUFS; None if silent.
        gain: Gain the step applies, dB; None if silent.
    """

    step: str
    loudness: float | None
    gain: float | None


def step_filter(step: Step, gain_db: float | None = None) -> str:
    """One step as filter text.

    A filter becomes ``type`` or ``type=k=v:k=v`` (values quoted). A loudness
    step becomes the gain it measured, or ``anull`` when the track is silent.
    """
    if step.type == "loudness":
        return "anull" if gain_db is None else f"volume={gain_db:.2f}dB"
    if not step.options:
        return step.type
    args = ":".join(f"{k}={quote(v)}" for k, v in step.options.items())
    return f"{step.type}={args}"


def chain_filter(steps: Sequence[Step], gains: Gains, upto: int | None = None) -> str:
    """The chain ``steps[:upto]`` as one filter text, starting in float.

    Raises:
        ValueError: If a loudness step has no entry in `gains` (it must be
            measured first).
    """
    parts = [FLOAT]
    for i, step in enumerate(steps[:upto]):
        if step.type == "loudness":
            if i not in gains:
                raise ValueError(f"{step.where or step.name}: loudness not measured")
            parts.append(step_filter(step, gains[i]))
        else:
            parts += [step_filter(step), DENORM]
            if step.type == "firequalizer":
                parts.append(_undelay(step))
    return ",".join(parts)


def _undelay(step: Step) -> str:
    """Cut firequalizer's latency from the start of the track.

    The eq's output lags its input by its ``delay``; without the cut every
    voice with an eq would play that much later than the music.
    """
    delay = step.options.get("delay", EQ_DELAY)
    return f"atrim=start={quote(delay)},asetpts=PTS-STARTPTS"


def measure_filter(steps: Sequence[Step], gains: Gains, index: int) -> str:
    """The chain before the loudness step at `index`, then ``ebur128``."""
    dual = quote(steps[index].options.get("dual_mono", True))
    # framelog=verbose keeps the per-frame lines out of stderr (info level)
    meter = f"ebur128=dualmono={dual}:framelog=verbose"
    return f"{chain_filter(steps, gains, index)},{meter}"


def parse_integrated(stderr: str) -> float | None:
    """Integrated loudness from ebur128's summary; None if silent or absent."""
    _, found, summary = stderr.rpartition("Summary:")
    match = _INTEGRATED.search(summary) if found else None
    if match is None or "inf" in match.group(1):
        return None
    value = float(match.group(1))
    return None if value <= SILENT_LUFS else value


def measure(inputs: list[str], graph: str, label: str) -> float | None:
    """Run `graph` (ending in ebur128) on `inputs`; the integrated loudness."""
    done = ffmpeg.run([*inputs, "-filter_complex", graph, "-f", "null", "-"], label)
    return parse_integrated(done.stderr)


def track_gains(
    path: Path, prep: str, steps: Sequence[Step], label: str
) -> tuple[Gains, list[Measured]]:
    """Measure the loudness steps of one track's chain, in order.

    Each measurement runs the chain up to that step, with the gains measured
    so far. Gain = the step's target minus the measured loudness.

    Args:
        path: The track.
        prep: Input filter for the track (channel layout).
        steps: The track's chain.
        label: Track name for error messages.
    """
    gains: Gains = {}
    measured = []
    for i, step in enumerate(steps):
        if step.type != "loudness":
            continue
        graph = f"[0:a]{prep},{measure_filter(steps, gains, i)}"
        level = measure(["-i", str(path)], graph, f"measure {label}, {step.name}")
        gain = None if level is None else float(step.options["target"]) - level
        gains[i] = gain
        measured.append(Measured(step.name, level, gain))
    return gains, measured
