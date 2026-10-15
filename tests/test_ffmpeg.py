"""Shared ffmpeg helpers: lookup, version, runs, decoding, duration."""

import sys
import threading
import time

import pytest

from scribe_rpg import ffmpeg
from scribe_rpg.errors import ScribeError

from helpers import make_flac


def test_require_raises_hint_when_missing(monkeypatch):
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: None)
    with pytest.raises(ScribeError, match="ffmpeg not found"):
        ffmpeg.require()


def test_require_ok_when_found(monkeypatch):
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: "ffmpeg")
    ffmpeg.require()


def test_version_parsing(monkeypatch):
    out = ffmpeg.subprocess.CompletedProcess(
        [], 0, stdout="ffmpeg version 7.1.1 Copyright (c) 2000-2025 x\n"
    )
    monkeypatch.setattr(ffmpeg.subprocess, "run", lambda *a, **k: out)
    assert ffmpeg.version() == "7.1.1"


def test_version_unknown_on_odd_output(monkeypatch):
    out = ffmpeg.subprocess.CompletedProcess([], 0, stdout="something else\n")
    monkeypatch.setattr(ffmpeg.subprocess, "run", lambda *a, **k: out)
    assert ffmpeg.version() == "unknown"


def test_version_unknown_when_ffmpeg_fails(monkeypatch):
    def boom(*a, **k):
        raise OSError("no ffmpeg")

    monkeypatch.setattr(ffmpeg.subprocess, "run", boom)
    assert ffmpeg.version() == "unknown"


@pytest.mark.parametrize("found", ["4.4.2", "6.0.1"])
def test_version_warning_for_old_ffmpeg(monkeypatch, found):
    monkeypatch.setattr(ffmpeg, "version", lambda: found)
    assert ffmpeg.version_warning() == (
        f"ffmpeg {found} is older than 6.1, the oldest scribe supports; "
        "mastering may fail"
    )


def test_install_hint_has_no_version_cap():
    # the "<8" cap was torchcodec's (WhisperX), not scribe's; mastering and
    # transcription both ran on ffmpeg 9
    assert "<8" not in ffmpeg.HINT and "6.1 or newer" in ffmpeg.HINT
    assert "libmp3lame" in ffmpeg.HINT


@pytest.mark.parametrize("found", ["6.1", "6.1.0", "7.1", "N-11234-gabc", "unknown"])
def test_no_version_warning(monkeypatch, found):
    monkeypatch.setattr(ffmpeg, "version", lambda: found)
    assert ffmpeg.version_warning() is None


@pytest.mark.ffmpeg
def test_run_error_names_label_and_stderr(tmp_path):
    missing = tmp_path / "missing.flac"
    with pytest.raises(ScribeError) as err:
        ffmpeg.run(["-i", str(missing), "-f", "null", "-"], "measure x")
    text = str(err.value)
    assert text.startswith("ffmpeg failed (measure x):")
    assert "missing.flac" in text


@pytest.mark.ffmpeg
def test_run_returns_output(tmp_path):
    flac = make_flac(tmp_path / "a.flac", 0.5)
    done = ffmpeg.run(["-i", str(flac), "-f", "null", "-"], "read")
    assert done.returncode == 0


@pytest.mark.ffmpeg
def test_decode_chunks(tmp_path):
    flac = make_flac(tmp_path / "a.flac", 2.5)
    chunks = list(ffmpeg.decode(flac, 8000, chunk_seconds=1))
    sizes = [len(c) for c in chunks]
    assert sizes[:2] == [8000, 8000]
    assert len(sizes) == 3 and abs(sizes[2] - 4000) < 100
    assert all(c.dtype.name == "float32" for c in chunks)


@pytest.mark.ffmpeg
def test_decode_error(tmp_path):
    bad = tmp_path / "bad.flac"
    bad.write_bytes(b"not audio")
    with pytest.raises(ScribeError, match=r"ffmpeg failed \(decode bad.flac\)"):
        list(ffmpeg.decode(bad, 8000))


@pytest.mark.ffmpeg
def test_duration(tmp_path):
    flac = make_flac(tmp_path / "a.flac", 2)
    assert ffmpeg.duration(flac) == pytest.approx(2.0, abs=0.05)


@pytest.mark.ffmpeg
def test_duration_ends_ffmpeg_normally(tmp_path, monkeypatch):
    # ffmpeg 7.1 sometimes deadlocks on the "no output file" error exit
    # (about 1 in 7000 launches here), so the duration read gives an output
    codes = []
    attempt = ffmpeg.attempt

    def recording(args):
        done = attempt(args)
        codes.append(done.returncode)
        return done

    monkeypatch.setattr(ffmpeg, "attempt", recording)
    ffmpeg.duration(make_flac(tmp_path / "a.flac", 2))
    assert codes == [0]


@pytest.mark.ffmpeg
def test_duration_of_a_non_audio_file(tmp_path):
    bad = tmp_path / "bad.flac"
    bad.write_bytes(b"not audio")
    with pytest.raises(ScribeError, match=r"ffmpeg failed \(duration bad.flac\)"):
        ffmpeg.duration(bad)


@pytest.mark.ffmpeg
def test_make_flac_on_ranges(tmp_path):
    import numpy as np

    flac = make_flac(tmp_path / "a.flac", 3, on=[(1, 2)])
    audio = np.concatenate(list(ffmpeg.decode(flac, 8000)))
    rms = [
        float(np.sqrt(np.mean(audio[i * 8000 : (i + 1) * 8000] ** 2))) for i in range(3)
    ]
    # volume switches per frame (~21 ms), so the edges may leak a little
    assert rms[0] < 0.02 and rms[2] < 0.02 and rms[1] > 0.05


@pytest.mark.ffmpeg
def test_run_progress_returns_stderr():
    seen = []
    stderr = ffmpeg.run_progress(
        ["-f", "lavfi", "-i", "sine=d=2", "-af", "ebur128=framelog=verbose",
         "-f", "null", "-"],
        "meter", seen.append,
    )  # fmt: skip
    assert "Summary:" in stderr
    assert seen and seen[-1] == pytest.approx(2, abs=0.1)


# --- stopping: errors, Ctrl+C, chatty ffmpeg ---------------------------------


def fake_ffmpeg(monkeypatch, script):
    """Run `script` (Python) in place of ffmpeg; ffmpeg's arguments follow it."""
    monkeypatch.setattr(ffmpeg, "_BASE", [sys.executable, "-c", script])


def exit_with(code):
    """A script that exits with `code`; Python 3.10's sys.exit cannot give
    Windows codes above 2**31 (it exits with 0xffffffff), ExitProcess can."""
    if code > 255 and sys.platform == "win32":
        return f"import ctypes; ctypes.windll.kernel32.ExitProcess({code})"
    return f"import sys; sys.exit({code})"


def test_decode_with_a_flood_of_errors_does_not_hang(monkeypatch, tmp_path):
    # a damaged FLAC: one error line per bad frame, far more than a pipe holds
    fake_ffmpeg(
        monkeypatch,
        "import sys; sys.stderr.write(('bad frame' + chr(10)) * 200_000); "
        "sys.stdout.buffer.write(bytes(64))",
    )
    chunks = []
    worker = threading.Thread(
        target=lambda: chunks.extend(ffmpeg.decode(tmp_path / "x.flac", 16000)),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=20)
    assert not worker.is_alive(), "decode hung"
    assert sum(len(c) for c in chunks) == 16


@pytest.mark.parametrize("code", [255, 0xC000013A])
def test_ffmpeg_stopped_by_ctrl_c_is_an_interrupt(monkeypatch, code):
    # Ctrl+C reaches ffmpeg too: 255 ("received signal"), or Windows' code
    if code > 255 and sys.platform != "win32":
        pytest.skip("exit codes above 255 exist only on Windows")
    fake_ffmpeg(monkeypatch, exit_with(code))
    with pytest.raises(KeyboardInterrupt):
        ffmpeg.run([], "test")
    with pytest.raises(KeyboardInterrupt):
        ffmpeg.run_progress([], "test")


def test_parallel_keeps_order():
    assert ffmpeg.parallel(lambda x: x * 2, [3, 1, 2]) == [6, 2, 4]


SLOW = ["-re", "-f", "lavfi", "-i", "anullsrc", "-t", "60", "-f", "null", "-"]


@pytest.mark.ffmpeg
@pytest.mark.parametrize("error", [ScribeError("boom"), KeyboardInterrupt()])
def test_parallel_stops_at_the_first_error(monkeypatch, error):
    # 2 workers, 6 items: item 0 fails while item 1 runs a 60 s ffmpeg;
    # the queue must not start and the running ffmpeg must be killed
    monkeypatch.setattr(ffmpeg, "workers", lambda: 2)
    started = []

    def work(i):
        started.append(i)
        if i == 0:
            time.sleep(0.5)
            raise error
        ffmpeg.run(SLOW, f"slow {i}")

    begin = time.monotonic()
    with pytest.raises(type(error)):
        ffmpeg.parallel(work, list(range(6)))
    assert time.monotonic() - begin < 10
    assert sorted(started) == [0, 1]
    ffmpeg.run(["-f", "lavfi", "-i", "anullsrc", "-t", "0.1", "-f", "null", "-"],
               "after")  # fmt: skip


def test_silent_crash_names_the_exit_code(monkeypatch):
    # a crashing ffmpeg prints nothing; the exit code is the only clue
    code = 0xC0000005 if sys.platform == "win32" else 139
    fake_ffmpeg(monkeypatch, exit_with(code))
    shown = f"{code:#x}" if code > 255 else str(code)
    with pytest.raises(ScribeError, match=f"no error output, exit code {shown}"):
        ffmpeg.run([], "test")
