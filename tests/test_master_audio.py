"""Loudness measurement and filters on real (synthetic) audio through ffmpeg."""

import pytest

from scribe_rpg.master.catalogue import Step
from scribe_rpg.master.render import (
    chain_filter,
    measure,
    measure_filter,
    track_gains,
)

from helpers import make_flac

pytestmark = pytest.mark.ffmpeg

MONO = "aformat=channel_layouts=mono"


def loudness(path, steps, gains, index):
    graph = f"[0:a]{MONO},{measure_filter(steps, gains, index)}"
    return measure(["-i", str(path)], graph, "test")


def test_loudness_step_reaches_its_target(tmp_path):
    tone = make_flac(tmp_path / "a.flac", 5, volume_db=-30)
    steps = [
        Step("level", "loudness", {"target": -19}),
        Step("check", "loudness", {"target": -19}),
    ]
    gains, measured = track_gains(tone, MONO, steps, "a")
    assert measured[0].gain == pytest.approx(-19 - measured[0].loudness)
    assert measured[1].loudness == pytest.approx(-19, abs=0.1)
    assert gains[1] == pytest.approx(0, abs=0.1)


def test_dual_mono_off_measures_3_db_lower(tmp_path):
    tone = make_flac(tmp_path / "a.flac", 5, volume_db=-30)
    dual = [Step("l", "loudness", {"target": -19})]
    single = [Step("l", "loudness", {"target": -19, "dual_mono": False})]
    difference = loudness(tone, dual, {}, 0) - loudness(tone, single, {}, 0)
    assert difference == pytest.approx(3.0, abs=0.1)


def test_silent_track_gets_no_gain(tmp_path):
    silent = make_flac(tmp_path / "a.flac", 3, on=[])
    gains, [m] = track_gains(
        silent, MONO, [Step("l", "loudness", {"target": -19})], "a"
    )
    assert gains == {0: None} and m.loudness is None and m.gain is None
    assert chain_filter([Step("l", "loudness", {})], gains).endswith(",anull")


@pytest.mark.parametrize(("freq", "low", "high"), [(30, 6, 99), (1000, -1, 1)])
def test_highpass_80(tmp_path, freq, low, high):
    tone = make_flac(tmp_path / "a.flac", 5, freq=freq, volume_db=-12)
    meter = Step("m", "loudness", {"target": -19})
    hp = Step("rumble", "highpass", {"f": 80})
    before = loudness(tone, [meter], {}, 0)
    after = loudness(tone, [hp, meter], {}, 1)
    assert low < before - after < high
