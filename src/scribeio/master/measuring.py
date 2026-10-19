"""Measuring with live progress: one row per track, a line when it is done.

The measured levels and gains are the mastering report; they are applied
either way and printed only when asked (``--verbose``). A silent track is
always a warning.
"""

from __future__ import annotations

from scribeio.master import mix
from scribeio.master.catalogue import Step
from scribeio.master.render import Gains, Measured
from scribeio.progress import Progress


def measure(
    plans: list[mix.TrackPlan],
    mix_steps: list[Step],
    channels: str,
    progress: Progress,
    verbose: bool = False,
) -> Gains:
    """`mix.measure_all` with a live row per track (and ``mix``, if its chain
    has a loudness step); prints ``✔ measure <account>`` as each finishes.

    Returns:
        The mix gains (``plan.gains`` are filled).
    """
    rows = [p.track.account for p in plans]
    rows += ["mix"] if any(s.type == "loudness" for s in mix_steps) else []
    n = len(plans)
    with progress.rows(rows, f"measure {n} track{'s' if n != 1 else ''}") as live:

        def show(row: str, measured: list[Measured]) -> None:
            parts = []
            for m in measured:
                if m.loudness is None or m.gain is None:
                    progress.warn(f"{row}: silent, {m.step} applies no gain")
                else:
                    parts.append(f"{m.step} {m.loudness:.1f} LUFS → {m.gain:+.1f} dB")
            detail = f": {', '.join(parts)}" if verbose and parts else ""
            live.done(row, f"measure {row}{detail}")

        return mix.measure_all(plans, mix_steps, channels, show, live.update)
