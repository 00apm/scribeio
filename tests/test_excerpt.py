"""Excerpt finder: activity per second, window and montage choice (pure)."""

import numpy as np
import pytest

from scribe_rpg.errors import ScribeError
from scribe_rpg.master.excerpt import (
    Excerpt,
    activities,
    activity,
    choose,
    per_second,
)

from helpers import make_flac


def act(*parts):
    """Activity per second from (seconds, share) pairs."""
    return np.concatenate([np.full(n, v, dtype=np.float64) for n, v in parts])


# --- per_second -------------------------------------------------------------


def test_per_second_shares_of_active_frames():
    # 50 ms frames: 20 per second; second 0 half active, second 1 silent
    levels = np.array([-20.0] * 10 + [-90.0] * 10 + [-90.0] * 20)
    assert per_second(levels, 50, -50).tolist() == [0.5, 0.0]


def test_per_second_keeps_last_partial_second():
    levels = np.array([-90.0] * 20 + [-20.0] * 5)
    assert per_second(levels, 50, -50).tolist() == [0.0, 1.0]


def test_per_second_empty():
    assert len(per_second(np.zeros(0), 50, -50)) == 0


# --- window -----------------------------------------------------------------


def test_least_active_voice_decides():
    # window A (0-10): a speaks all the time, b is silent
    # window B (10-20): both speak a little
    a = act((10, 1.0), (10, 0.2))
    b = act((10, 0.0), (10, 0.2))
    ex = choose({"a": a, "b": b}, [], 10)
    assert ex == Excerpt([(10, 20)], montage=False, missing=[])


def test_music_breaks_ties():
    a = act((30, 0.5))
    music = act((20, 0.0), (10, 1.0))
    assert choose({"a": a}, [music], 10).ranges == [(20, 30)]


def test_earliest_on_full_tie():
    a = act((30, 0.5))
    assert choose({"a": a}, [], 10).ranges == [(0, 10)]


def test_tracks_of_different_lengths_are_padded():
    # b leaves after 10 s; a speaks from 5 s to the end at 30 s
    a = act((5, 0.0), (25, 1.0))
    b = act((10, 1.0))
    ex = choose({"a": a, "b": b}, [], 5)
    assert ex.ranges == [(5, 10)] and not ex.montage


def test_recording_shorter_than_length():
    a = act((40, 0.3))
    assert choose({"a": a}, [], 90).ranges == [(0, 40)]


def test_ranges_end_at_the_padded_length():
    a = act((20, 0.3))
    music = act((50, 0.0))
    assert choose({"a": a}, [music], 90).ranges == [(0, 50)]


# --- montage ----------------------------------------------------------------


def test_no_common_window_gives_montage():
    # a speaks only 0-20, b only 60-80: no 30 s window holds both
    a = act((20, 1.0), (80, 0.0))
    b = act((60, 0.0), (20, 1.0), (20, 0.0))
    ex = choose({"a": a, "b": b}, [], 30)
    assert ex.montage
    assert ex.ranges == [(0, 15), (60, 75)]
    assert sum(e - s for s, e in ex.ranges) <= 30
    assert ex.missing == []


def test_montage_merges_overlapping_pieces():
    a = act((10, 0.0), (10, 1.0), (80, 0.0))
    b = act((15, 0.0), (10, 1.0), (75, 0.0))
    c = act((100, 0.0))
    ex = choose({"a": a, "b": b, "c": c}, [], 30, montage=True)
    # pieces of 10 s: a 10-20, b 15-25, c (silent) earliest 0-10
    assert ex.ranges == [(0, 25)]
    assert ex.missing == ["c"]


def test_silent_voice_is_missing():
    a = act((60, 1.0))
    b = act((60, 0.0))
    ex = choose({"a": a, "b": b}, [], 20)
    assert ex.montage and ex.missing == ["b"]


def test_forced_montage():
    a = act((60, 1.0))
    b = act((30, 0.0), (30, 1.0))
    ex = choose({"a": a, "b": b}, [], 20, montage=True)
    assert ex.montage
    assert ex.ranges == [(0, 10), (30, 40)]


# --- at ---------------------------------------------------------------------


def test_at_inside():
    a = act((100, 0.1))
    assert choose({"a": a}, [], 30, at=80) == Excerpt([(80, 100)], False, [])


def test_at_past_the_end():
    a = act((3725, 0.1))
    with pytest.raises(ScribeError) as err:
        choose({"a": a}, [], 30, at=3725)
    assert str(err.value) == "--at 01:02:05 is past the end (01:02:05)"


def test_empty_recording():
    with pytest.raises(ScribeError, match="no audio"):
        choose({"a": np.zeros(0)}, [], 30)


# --- activity from audio (ffmpeg) -------------------------------------------

EXPECTED = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0]


@pytest.mark.ffmpeg
@pytest.mark.parametrize("channels", [1, 2])
def test_activity_from_flac(tmp_path, channels):
    flac = make_flac(tmp_path / "a.flac", 10, on=[(2, 4), (7, 8)], channels=channels)
    assert activity(flac, 50, -50).tolist() == pytest.approx(EXPECTED, abs=0.05)


@pytest.mark.ffmpeg
def test_activities_keep_order(tmp_path):
    loud = make_flac(tmp_path / "1-loud.flac", 3)
    quiet = make_flac(tmp_path / "2-quiet.flac", 3, on=[])
    first, second = activities([loud, quiet], 50, -50)
    assert first.tolist() == pytest.approx([1, 1, 1])
    assert second.tolist() == [0, 0, 0]
