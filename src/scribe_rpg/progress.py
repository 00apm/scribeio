"""Console progress on stderr, shared by all commands.

One line per finished step with a check mark and the elapsed time. While a
step runs, a spinner with a ticking clock shows that work is going on. Tracks
get a bar that also moves within a track and names the current phase;
work running in parallel gets one live row per item; downloads get a bar
counting bytes. Warnings are printed as they happen.
Text is printed literally: square brackets in names or timestamps are never
read as rich markup.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.progress import Progress as RichProgress


class Progress:
    """Progress reporter for one command run.

    Attributes:
        console: Where everything is printed (stderr by default).
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(stderr=True)

    @contextmanager
    def step(self, label: str) -> Iterator[None]:
        """Show a spinner and elapsed time while the block runs.

        Afterwards prints ``✔ label (1.2s)``. Nothing is printed if the
        block raises; the error is reported by the caller.
        """
        t0 = time.monotonic()
        columns = (
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
        )
        with RichProgress(*columns, console=self.console, transient=True) as spinner:
            spinner.add_task(escape(label), total=None)
            yield
        self.done(label, time.monotonic() - t0)

    @contextmanager
    def busy(self, label: str) -> Iterator[None]:
        """Show a spinner while the block runs; leave nothing behind.

        For work whose result is printed afterwards (``scribe check``).
        Nothing is shown when the console is not a terminal.
        """
        with self.console.status(escape(label)):
            yield

    def done(self, label: str, seconds: float) -> None:
        """Print a finished line: ``✔ label (12.3s)``."""
        self.console.print(f"[green]✔[/] {escape(label)} [dim]({seconds:.1f}s)[/]")

    def warn(self, message: str) -> None:
        """Print a warning."""
        self.console.print(f"[yellow]⚠[/] {escape(message)}")

    @contextmanager
    def download(
        self, label: str, total: int | None
    ) -> Iterator[Callable[[int], None]]:
        """Show a byte-counting bar; yields ``advance(n_bytes)``.

        Args:
            label: What is downloaded, e.g. a file name.
            total: Expected size in bytes, or None if unknown.
        """
        t0 = time.monotonic()
        columns = (
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
        )
        with RichProgress(*columns, console=self.console, transient=True) as bar:
            task = bar.add_task(escape(f"download {label}"), total=total)
            yield lambda n: bar.advance(task, n)
        self.done(f"download {label}", time.monotonic() - t0)

    @contextmanager
    def rows(self, labels: list[str], label: str) -> Iterator[Rows]:
        """One live row per label for work running in parallel.

        Each row shows its name, phase, a bar, percent and elapsed time; rows
        start as ``waiting``. Afterwards prints ``✔ label (12.3s)``; nothing
        if the block raises.
        """
        t0 = time.monotonic()
        columns = (
            TextColumn("  {task.description}"),
            TextColumn("{task.fields[phase]}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        with RichProgress(*columns, console=self.console, transient=True) as live:
            yield Rows(self, live, labels)
        self.done(label, time.monotonic() - t0)

    @contextmanager
    def count(self, live: str, label: str, total: int) -> Iterator[Callable[[], None]]:
        """Show a bar counting `total` items as ``live  M/N``; yields
        ``advance()``. Afterwards prints ``✔ label (12.3s)``."""
        t0 = time.monotonic()
        columns = (
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        with RichProgress(*columns, console=self.console, transient=True) as bar:
            task = bar.add_task(escape(live), total=total)
            yield lambda: bar.advance(task)
        self.done(label, time.monotonic() - t0)

    @contextmanager
    def bar(self, label: str) -> Iterator[Callable[[float], None]]:
        """Show a percent bar; yields ``update(percent)`` (0..100).
        Afterwards prints ``✔ label (12.3s)``."""
        t0 = time.monotonic()
        columns = (
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        with RichProgress(*columns, console=self.console, transient=True) as bar:
            task = bar.add_task(escape(label), total=100)
            yield lambda percent: bar.update(task, completed=percent)
        self.done(label, time.monotonic() - t0)

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


class Rows:
    """Handle for the live rows; see `Progress.rows`. Safe to call from
    worker threads (rich locks its own updates)."""

    def __init__(self, progress: Progress, live: RichProgress, labels: list[str]):
        self._progress = progress
        self._live = live
        self._tasks = {
            row: live.add_task(escape(row), total=100, start=False, phase="waiting")
            for row in labels
        }

    def update(self, row: str, phase: str, percent: float) -> None:
        """Show `row`'s current phase and percent done (starts its clock)."""
        task = self._tasks.get(row)
        if task is None:
            return
        self._live.start_task(task)  # only the first call starts the clock
        # below 100: rich would freeze the clock of a finished task, and the
        # row's next pass starts again at 0 (still shown as 100%)
        completed = min(percent, 99.9)
        self._live.update(task, phase=escape(phase), completed=completed)

    def done(self, row: str, text: str) -> None:
        """Remove `row` and print ``✔ text (time)`` with its elapsed time."""
        task = self._tasks.pop(row, None)
        if task is None:
            return
        elapsed = next(t.elapsed for t in self._live.tasks if t.id == task)
        self._live.remove_task(task)
        self._progress.done(text, elapsed or 0.0)


class TrackBar:
    """Handle for the tracks bar; see `Progress.tracks`."""

    def __init__(self, progress: Progress, bar: RichProgress, task: TaskID) -> None:
        self._progress = progress
        self._bar = bar
        self._task = task
        self._account = ""
        self._phase = ""
        self._finished = 0
        self._percent = 0.0
        self._started = time.monotonic()

    @property
    def position(self) -> float:
        """Finished tracks plus the fraction of the current one."""
        return self._finished + self._percent / 100

    @property
    def phase(self) -> str:
        """What the current track is going through, e.g. ``transcribing``."""
        return self._phase

    def set_phase(self, phase: str) -> None:
        """Show a new phase for the current track (told by the transcriber)."""
        self._phase = phase
        self._refresh()

    def start(self, account: str) -> None:
        """Begin a track: show its name and phase next to the bar."""
        self._account = account
        self._phase = "starting"
        self._percent = 0.0
        self._started = time.monotonic()
        self._refresh()

    def update(self, percent: float) -> None:
        """Move the bar within the current track (0..100)."""
        self._percent = max(0.0, min(percent, 100.0))
        self._refresh()

    def done(self, label: str) -> None:
        """Finish the current track and print ``✔ label (time)``."""
        self._progress.done(label, time.monotonic() - self._started)
        self._finished += 1
        self._percent = 0.0
        self._bar.update(self._task, completed=self.position)

    def _refresh(self) -> None:
        description = escape(f"{self._account}: {self.phase}")
        self._bar.update(self._task, description=description, completed=self.position)
