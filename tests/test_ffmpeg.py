"""Shared ffmpeg helpers: lookup, version, runs, decoding, duration."""

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


def test_version_warning_for_old_ffmpeg(monkeypatch):
    monkeypatch.setattr(ffmpeg, "version", lambda: "4.3.1")
    warning = ffmpeg.version_warning()
    assert warning is not None and "4.3.1" in warning


@pytest.mark.parametrize("found", ["7.1", "N-11234-gabc", "unknown"])
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
