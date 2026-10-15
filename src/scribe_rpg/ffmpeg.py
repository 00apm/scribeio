"""The ffmpeg command line: lookup, version, runs, decoding, duration.

Shared by ``scribe transcribe`` and ``scribe master``. ffmpeg is always called
with an argument list, never through a shell.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, TYPE_CHECKING

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


def run(args: list[str], label: str) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg with `args`; capture stdout and stderr as text.

    Raises:
        ScribeError: If ffmpeg exits non-zero; names `label` and the last
            lines of its stderr.
    """
    done = subprocess.run(
        [*_BASE, *args], capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )  # fmt: skip
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
    """
    proc = subprocess.Popen(
        [*_BASE, *args, "-progress", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace",
    )  # fmt: skip
    assert proc.stdout is not None and proc.stderr is not None
    stderr = proc.stderr
    err: list[str] = []
    drain = threading.Thread(target=lambda: err.append(stderr.read()))
    drain.start()
    try:
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            # out_time_us is in microseconds; "N/A" before the first frame
            if key == "out_time_us" and on_time and value.lstrip("-").isdigit():
                on_time(max(0, int(value)) / 1_000_000)
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        drain.join()
        proc.stdout.close()
        proc.stderr.close()
    stderr_text = "".join(err)
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
    """
    import numpy as np

    args = [*_BASE, "-loglevel", "error", "-i", str(path), "-ac", "1",
            "-ar", str(rate), "-f", "f32le", "pipe:1"]  # fmt: skip
    size = rate * chunk_seconds * 4  # bytes per chunk (4 per float32 sample)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    try:
        while True:
            data = _read(proc.stdout, size)
            if not data:
                break
            usable = len(data) - len(data) % 4
            yield np.frombuffer(data[:usable], dtype=np.float32)
        err = proc.stderr.read().decode("utf-8", "replace")
        if proc.wait() != 0:
            raise ScribeError(f"ffmpeg failed (decode {path.name}): {_tail(err)}")
    finally:
        # also reached when the caller stops early: do not leave ffmpeg running
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        proc.stderr.close()


def duration(path: Path) -> float:
    """Seconds, from the ``Duration:`` line that ``ffmpeg -i`` prints.

    Raises:
        ScribeError: If ffmpeg prints no duration (not an audio file).
    """
    # `ffmpeg -i` without an output exits 1 by design, so no `run` here.
    done = subprocess.run(
        [*_BASE, "-i", str(path)], capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )  # fmt: skip
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
