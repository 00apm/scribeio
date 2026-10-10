"""Console progress on stderr, shared by all commands.

One line per finished step with a check mark and the elapsed time, a bar
over tracks that also moves within a track, and warnings printed as they
happen. Text is printed literally: square brackets in names or timestamps
are never read as rich markup.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.progress import Progress as RichProgress


class Progress:
    """Progress reporter for one command run.

    Attributes:
        console: Where everything is printed (stderr by default).
        warnings: Every warning printed so far.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(stderr=True)
        self.warnings: list[str] = []

    @contextmanager
    def step(self, label: str) -> Iterator[None]:
        """Show a spinner while the block runs, then ``✔ label (1.2s)``.

        Nothing is printed if the block raises; the error is reported by the
        caller.
        """
        t0 = time.monotonic()
        with self.console.status(escape(label)):
            yield
        self.done(label, time.monotonic() - t0)

    def done(self, label: str, seconds: float) -> None:
        """Print a finished line: ``✔ label (12.3s)``."""
        self.console.print(f"[green]✔[/] {escape(label)} [dim]({seconds:.1f}s)[/]")

    def warn(self, message: str) -> None:
        """Print a warning and remember it."""
        self.warnings.append(message)
        self.console.print(f"[yellow]⚠[/] {escape(message)}")

    @contextmanager
    def tracks(self, total: int) -> Iterator[TrackBar]:
        """Show a bar over `total` tracks while the block runs."""
        columns = (
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        with RichProgress(*columns, console=self.console) as bar:
            task = bar.add_task("transcribing", total=total)
            yield TrackBar(self, bar, task)


class TrackBar:
    """Handle for the tracks bar; see `Progress.tracks`."""

    def __init__(self, progress: Progress, bar: RichProgress, task: TaskID) -> None:
        self._progress = progress
        self._bar = bar
        self._task = task
        self._finished = 0
        self._percent = 0.0
        self._started = time.monotonic()

    @property
    def position(self) -> float:
        """Finished tracks plus the fraction of the current one."""
        return self._finished + self._percent / 100

    def start(self, account: str) -> None:
        """Begin a track: show its name next to the bar."""
        self._percent = 0.0
        self._started = time.monotonic()
        description = escape(f"transcribing {account}")
        self._bar.update(self._task, description=description, completed=self.position)

    def update(self, percent: float) -> None:
        """Move the bar within the current track (0..100)."""
        self._percent = max(0.0, min(percent, 100.0))
        self._bar.update(self._task, completed=self.position)

    def done(self, label: str) -> None:
        """Finish the current track and print ``✔ label (time)``."""
        self._progress.done(label, time.monotonic() - self._started)
        self._finished += 1
        self._percent = 0.0
        self._bar.update(self._task, completed=self.position)
