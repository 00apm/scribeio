"""The ffmpeg command line: lookup, version, runs, decoding, duration.

Shared by ``scribe transcribe`` and ``scribe master``. ffmpeg is always called
with an argument list, never through a shell.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, TypeVar

from scribe_rpg.errors import ScribeError

if TYPE_CHECKING:
    # numpy only inside decode: `scribe merge` imports this module too
    import numpy as np
    import numpy.typing as npt

HINT = (
    'ffmpeg not found in PATH; install it (conda install -c conda-forge "ffmpeg<8", '
    "winget install ffmpeg, apt install ffmpeg or brew install ffmpeg)"
)
# amix normalize first appears in 4.4 (absent in 4.3's filter docs);
# ebur128 dualmono and alimiter level already exist in 4.4.
MIN_VERSION = (4, 4)

_BASE = ["ffmpeg", "-hide_banner", "-nostdin", "-nostats"]
# How ffmpeg ends when Ctrl+C reaches it too: 255 after "received signal";
# Windows' STATUS_CONTROL_C_EXIT if it is stopped before it can handle it.
_CTRL_C_CODES = (255, 0xC000013A)

T = TypeVar("T")
R = TypeVar("R")

# Every running ffmpeg, so `parallel` can kill them all on the first error;
# no new ffmpeg starts while `_stopping` is set.
_live: set[subprocess.Popen[Any]] = set()
_lock = threading.Lock()
_stopping = threading.Event()


def require() -> None:
    """Raise a ScribeError with install instructions if ffmpeg is not in PATH."""
    if shutil.which("ffmpeg") is None:
        raise ScribeError(HINT)


def version() -> str:
    """Version from the first line of ``ffmpeg -version``, or ``"unknown"``."""
    try:
        run = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    parts = (run.stdout or "").split()
    if parts[:2] == ["ffmpeg", "version"] and len(parts) > 2:
        return parts[2]
    return "unknown"


def version_warning() -> str | None:
    """A warning if ffmpeg is older than MIN_VERSION; None if newer or unparsable.

    Git builds (``N-11234-gabc``) and ``unknown`` give no warning.
    """
    found = version()
    m = re.match(r"(\d+)\.(\d+)", found)
    if m is None or (int(m.group(1)), int(m.group(2))) >= MIN_VERSION:
        return None
    need = ".".join(map(str, MIN_VERSION))
    return f"ffmpeg {found} is older than {need}; mastering filters may fail"


def attempt(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg with `args`, whatever its exit code; stdout, stderr as text.

    Raises:
        KeyboardInterrupt: If ffmpeg was stopped by Ctrl+C.
    """
    proc = _start(
        [*_BASE, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )  # fmt: skip
    try:
        out, err = proc.communicate()
    finally:
        _end(proc)
    _check_interrupted(proc.returncode)
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def run(args: list[str], label: str) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg with `args`; capture stdout and stderr as text.

    Raises:
        ScribeError: If ffmpeg exits non-zero; names `label` and the last
            lines of its stderr.
        KeyboardInterrupt: If ffmpeg was stopped by Ctrl+C.
    """
    done = attempt(args)
    if done.returncode != 0:
        raise ScribeError(f"ffmpeg failed ({label}): {_tail(done.stderr)}")
    return done


def run_progress(
    args: list[str], label: str, on_time: Callable[[float], None] | None = None
) -> str:
    """Run ffmpeg like `run`, telling `on_time` the seconds written so far.

    Reads ffmpeg's ``-progress pipe:1`` lines; stderr is drained by a thread
    so a chatty ffmpeg cannot block. ffmpeg is killed if this is interrupted.

    Returns:
        ffmpeg's stderr (filters such as ebur128 report there).

    Raises:
        ScribeError: If ffmpeg exits non-zero; names `label` and the last
            lines of its stderr.
        KeyboardInterrupt: If ffmpeg was stopped by Ctrl+C.
    """
    proc = _start(
        [*_BASE, *args, "-progress", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace",
    )  # fmt: skip
    assert proc.stdout is not None and proc.stderr is not None
    stderr = _drain(proc.stderr)
    try:
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            # out_time_us is in microseconds; "N/A" before the first frame
            if key == "out_time_us" and on_time and value.lstrip("-").isdigit():
                on_time(max(0, int(value)) / 1_000_000)
        code = proc.wait()
    finally:
        _end(proc)
        stderr_text = stderr()
        proc.stdout.close()
        proc.stderr.close()
    _check_interrupted(code)
    if code != 0:
        raise ScribeError(f"ffmpeg failed ({label}): {_tail(stderr_text)}")
    return stderr_text


def decode(
    path: Path, rate: int, chunk_seconds: int = 60
) -> Iterator[npt.NDArray[np.float32]]:
    """Decode `path` to mono float32 at `rate`, in chunks of `chunk_seconds`.

    The last chunk is shorter. Memory stays at one chunk per track.

    Raises:
        ScribeError: If ffmpeg exits non-zero.
        KeyboardInterrupt: If ffmpeg was stopped by Ctrl+C.
    """
    import numpy as np

    args = [*_BASE, "-loglevel", "error", "-i", str(path), "-ac", "1",
            "-ar", str(rate), "-f", "f32le", "pipe:1"]  # fmt: skip
    size = rate * chunk_seconds * 4  # bytes per chunk (4 per float32 sample)
    proc = _start(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    # a damaged file makes ffmpeg log a line per bad frame: left unread, the
    # full stderr pipe would block ffmpeg while this waits for its stdout
    stderr = _drain(proc.stderr)
    try:
        while True:
            data = _read(proc.stdout, size)
            if not data:
                break
            usable = len(data) - len(data) % 4
            yield np.frombuffer(data[:usable], dtype=np.float32)
        code = proc.wait()
    finally:
        # also reached when the caller stops early: do not leave ffmpeg running
        _end(proc)
        err = stderr()
        proc.stdout.close()
        proc.stderr.close()
    _check_interrupted(code)
    if code != 0:
        raise ScribeError(f"ffmpeg failed (decode {path.name}): {_tail(err)}")


def duration(path: Path) -> float:
    """Seconds, from the ``Duration:`` line that ``ffmpeg -i`` prints.

    Raises:
        ScribeError: If ffmpeg prints no duration (not an audio file).
    """
    # `ffmpeg -i` without an output exits 1 by design, so no `run` here.
    done = attempt(["-i", str(path)])
    m = re.search(r"Duration: (\d+):(\d\d):(\d\d(?:\.\d+)?)", done.stderr)
    if m is None:
        raise ScribeError(f"ffmpeg failed (duration {path.name}): {_tail(done.stderr)}")
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def _read(stream: IO[bytes], size: int) -> bytes:
    """Read up to `size` bytes; fewer only at the end of the stream."""
    parts: list[bytes] = []
    left = size
    while left > 0:
        part = stream.read(left)
        if not part:
            break
        parts.append(part)
        left -= len(part)
    return b"".join(parts)


def _tail(stderr: str, lines: int = 5) -> str:
    """The last non-empty lines of ffmpeg's stderr, joined with " | "."""
    kept = [line.strip() for line in stderr.splitlines() if line.strip()]
    return " | ".join(kept[-lines:]) or "no error output"


def workers() -> int:
    """Threads for parallel ffmpeg runs: one per CPU this process may use
    (one ffmpeg graph uses about one core)."""
    affinity = getattr(os, "sched_getaffinity", None)  # Linux only
    return len(affinity(0)) if affinity else os.cpu_count() or 1


def parallel(work: Callable[[T], R], items: list[T]) -> list[R]:
    """`work` on every item in a thread pool; the results in order.

    On the first error or Ctrl+C, queued items never start, every running
    ffmpeg is killed, and the error is raised once the workers have ended.
    """

    def guarded(item: T) -> R:
        # a failing worker stops the others itself: it would otherwise take
        # the next queued item before the main thread can cancel the queue
        if _stopping.is_set():
            raise _Stopped
        try:
            return work(item)
        except BaseException:
            _stopping.set()
            raise

    pool = ThreadPoolExecutor(max_workers=workers())
    futures = [pool.submit(guarded, item) for item in items]
    try:
        pending = set(futures)
        while pending:
            # short waits: on Windows, Ctrl+C reaches a waiting main thread
            # only when its wait returns
            done, pending = wait(pending, timeout=0.2, return_when=FIRST_EXCEPTION)
            errors = [e for f in done if (e := f.exception()) is not None]
            errors = [e for e in errors if not isinstance(e, _Stopped)]
            if errors:
                stop = [e for e in errors if isinstance(e, KeyboardInterrupt)]
                raise (stop or errors)[0]
    except BaseException:
        _stopping.set()
        try:
            pool.shutdown(wait=False, cancel_futures=True)
            _kill_all()
            pool.shutdown(wait=True)
        finally:
            _stopping.clear()
        raise
    pool.shutdown()
    return [f.result() for f in futures]


class _Stopped(Exception):
    """Work not started because `parallel` is stopping after an error."""


def _start(args: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
    """Start ffmpeg and track it, unless `parallel` is stopping."""
    with _lock:
        if _stopping.is_set():
            raise _Stopped
        proc = subprocess.Popen(args, **kwargs)
        _live.add(proc)
    return proc


def _end(proc: subprocess.Popen[Any]) -> None:
    """Kill `proc` if it still runs and stop tracking it."""
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    with _lock:
        _live.discard(proc)


def _kill_all() -> None:
    with _lock:
        for proc in _live:
            if proc.poll() is None:
                proc.kill()


def _check_interrupted(code: int) -> None:
    """Raise KeyboardInterrupt if ffmpeg's exit `code` means Ctrl+C."""
    if code in _CTRL_C_CODES:
        raise KeyboardInterrupt


def _drain(stream: IO[Any]) -> Callable[[], str]:
    """Read `stream` to its end in a thread.

    Returns:
        A call that waits for the thread and returns the text read.
    """
    parts: list[str] = []

    def read() -> None:
        data = stream.read()
        parts.append(data if isinstance(data, str) else data.decode("utf-8", "replace"))

    thread = threading.Thread(target=read, daemon=True)
    thread.start()

    def text() -> str:
        thread.join()
        return "".join(parts)

    return text
