"""Console progress on stderr, shared by all commands.

One line per finished step with a check mark and the elapsed time. While a
step runs, a spinner with a ticking clock shows that work is going on. Tracks
get a bar that also moves within a track and names the current phase;
downloads get a bar counting bytes. Warnings are printed as they happen.
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
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.progress import Progress as RichProgress

# WhisperX reports 0-90 % for transcription and 90-100 % for alignment
# (see transcribe.whisper.Transcriber.transcribe).
_ALIGNING_FROM = 90.0


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

    def done(self, label: str, seconds: float) -> None:
        """Print a finished line: ``✔ label (12.3s)``."""
        self.console.print(f"[green]✔[/] {escape(label)} [dim]({seconds:.1f}s)[/]")

    def warn(self, message: str) -> None:
        """Print a warning and remember it."""
        self.warnings.append(message)
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
        self._account = ""
        self._finished = 0
        self._percent = 0.0
        self._started = time.monotonic()

    @property
    def position(self) -> float:
        """Finished tracks plus the fraction of the current one."""
        return self._finished + self._percent / 100

    @property
    def phase(self) -> str:
        """What the current track is going through, judged by its progress.

        WhisperX first detects speech (no progress reported yet), then
        transcribes, then aligns words.
        """
        if self._percent <= 0:
            return "voice detection"
        if self._percent < _ALIGNING_FROM:
            return "transcribing"
        return "aligning"

    def start(self, account: str) -> None:
        """Begin a track: show its name and phase next to the bar."""
        self._account = account
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
