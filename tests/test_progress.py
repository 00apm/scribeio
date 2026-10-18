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


def test_busy_shows_the_label_on_a_terminal_and_prints_no_check():
    buf = io.StringIO()
    p = Progress(Console(file=buf, width=120, force_terminal=True))
    with p.busy("checking"):
        pass
    assert "checking" in buf.getvalue()
    assert "✔" not in buf.getvalue()


def test_busy_prints_brackets_literally():
    buf = io.StringIO()
    p = Progress(Console(file=buf, width=120, force_terminal=True))
    with p.busy("checking [bold]x"):
        pass
    assert "checking [bold]x" in buf.getvalue()


def test_busy_leaves_stdout_and_stderr_alone():
    # Libraries imported under the spinner may keep sys.stdout (tqdm runs
    # colorama.init, which writes a reset code to it at exit); it must be
    # the real stream, not rich's proxy, or "[0m" shows up as text.
    import sys

    p = Progress(Console(file=io.StringIO(), force_terminal=True))
    before = sys.stdout, sys.stderr
    with p.busy("checking"):
        assert (sys.stdout, sys.stderr) == before


def test_busy_prints_nothing_when_not_a_terminal():
    p, buf = make()
    with p.busy("checking"):
        pass
    assert buf.getvalue() == ""


def test_progress_prints_brackets_literally():
    p, buf = make()
    p.warn("unknown account [bot]: name used [00:01:02]")
    with p.step("track [1/2]"):
        pass
    out = buf.getvalue()
    assert "⚠ unknown account [bot]: name used [00:01:02]" in out
    assert "track [1/2]" in out


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


def test_track_bar_shows_the_phase_it_is_told():
    p, _ = make()
    with p.tracks(1) as bar:
        bar.start("gm_account")
        assert bar.phase == "starting"
        bar.set_phase("loading Whisper [large-v3]")
        bar.update(10.0)  # progress does not change the phase
        assert bar.phase == "loading Whisper [large-v3]"
        bar.done("gm_account: 3 segments")
        bar.start("player_one")
        assert bar.phase == "starting"


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


def test_rows_print_a_line_per_finished_row_and_the_summary():
    p, buf = make()
    with p.rows(["gm_account", "player [1]", "mix"], "measure 2 tracks") as rows:
        rows.update("gm_account", "level 1/2", 50.0)
        rows.done("gm_account", "measure gm_account: level -20.0 LUFS → +1.0 dB")
        rows.update("player [1]", "loudness 2/2", 100.0)
        rows.done("player [1]", "measure player [1]")
    out = buf.getvalue()
    assert "✔ measure gm_account: level -20.0 LUFS → +1.0 dB (" in out
    assert "✔ measure player [1] (" in out
    assert "✔ measure 2 tracks (" in out
    assert "✔ mix" not in out  # never started, never done


def test_rows_failure_prints_no_summary():
    p, buf = make()
    with pytest.raises(RuntimeError):
        with p.rows(["a"], "measure 1 track"):
            raise RuntimeError("x")
    assert "✔" not in buf.getvalue()


def test_count_bar_done_line():
    p, buf = make()
    with p.count("preview: writing", "preview: write 2 files", 2) as advance:
        advance()
        advance()
    assert "✔ preview: write 2 files (" in buf.getvalue()


def test_percent_bar_done_line():
    p, buf = make()
    with p.bar("render s.mp3 (2 tracks + mix)") as update:
        update(50.0)
    assert "✔ render s.mp3 (2 tracks + mix) (" in buf.getvalue()


def test_row_clock_keeps_running_across_passes():
    # each pass runs 0..100 %; rich freezes a finished task's clock
    p, _ = make()
    with p.rows(["a"], "measure 1 track") as rows:
        rows.update("a", "l 1/2", 100.0)
        rows.update("a", "l2 2/2", 0.0)
        [task] = rows._live.tasks
        assert task.finished_time is None
