"""Filter text for ffmpeg, built from resolved steps (pure, no ffmpeg runs)."""

from __future__ import annotations

from collections.abc import Sequence

from scribe_rpg.master.catalogue import Step, quote

# Every chain starts in 32-bit float, so no step clips between steps.
FLOAT = "aformat=sample_fmts=flt"

Gains = dict[int, float | None]  # chain index of a loudness step -> dB (None: silent)


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
            parts.append(step_filter(step))
    return ",".join(parts)


def measure_filter(steps: Sequence[Step], gains: Gains, index: int) -> str:
    """The chain before the loudness step at `index`, then ``ebur128``."""
    dual = steps[index].options.get("dual_mono", True)
    return f"{chain_filter(steps, gains, index)},ebur128=dualmono={quote(dual)}"
