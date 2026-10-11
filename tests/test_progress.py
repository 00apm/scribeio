"""Tests for console progress output."""

import io

import pytest
from rich.console import Console

from scribe_rpg.progress import Progress


def make():
    buf = io.StringIO()
    return Progress(Console(file=buf, width=120, color_system=None)), buf


def test_step_prints_check_and_time():
    p, buf = make()
    with p.step("open session.zip"):
        pass
    assert "✔ open session.zip (" in buf.getvalue()


def test_step_failure_prints_no_check():
    p, buf = make()
    with pytest.raises(RuntimeError):
        with p.step("load model"):
            raise RuntimeError("x")
    assert "✔" not in buf.getvalue()


def test_progress_prints_brackets_literally():
    p, buf = make()
    p.warn("unknown account [bot]: name used [00:01:02]")
    with p.step("track [1/2]"):
        pass
    out = buf.getvalue()
    assert "⚠ unknown account [bot]: name used [00:01:02]" in out
    assert "track [1/2]" in out
    assert p.warnings == ["unknown account [bot]: name used [00:01:02]"]


def test_tracks_done_lines():
    p, buf = make()
    with p.tracks(2) as bar:
        bar.start("gm_account")
        bar.update(50.0)
        bar.update(100.0)
        bar.done("gm_account: 10 segments")
        bar.start("player_one")
        bar.done("player_one: 4 segments")
    out = buf.getvalue()
    assert "✔ gm_account: 10 segments" in out and "✔ player_one: 4 segments" in out


def test_track_bar_position_counts_finished_tracks_and_percent():
    p, _ = make()
    with p.tracks(4) as bar:
        bar.start("gm_account")
        bar.update(50.0)
        assert bar.position == pytest.approx(0.5)
        bar.done("gm_account: 1 segment")
        bar.update(25.0)
        assert bar.position == pytest.approx(1.25)


def test_track_bar_names_the_phase():
    p, _ = make()
    with p.tracks(1) as bar:
        bar.start("gm_account")
        assert bar.phase == "voice detection"  # before WhisperX reports progress
        bar.update(10.0)
        assert bar.phase == "transcribing"
        bar.update(95.0)
        assert bar.phase == "aligning"
        bar.done("gm_account: 3 segments")


def test_download_bar_counts_bytes():
    p, buf = make()
    with p.download("model.pth", total=300) as advance:
        advance(100)
        advance(200)
    assert "✔ download model.pth" in buf.getvalue()


def test_download_bar_failure_prints_no_check():
    p, buf = make()
    with pytest.raises(OSError):
        with p.download("model.pth", total=300) as advance:
            advance(100)
            raise OSError("connection reset")
    assert "✔" not in buf.getvalue()
