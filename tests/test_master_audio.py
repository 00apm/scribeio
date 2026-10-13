"""Loudness measurement and filters on real (synthetic) audio through ffmpeg."""

import re

import pytest

from scribe_rpg import ffmpeg
from scribe_rpg.errors import ScribeError
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


def test_eq_keeps_the_timing(tmp_path):
    # a tone from 1.0 s: after the eq it must still start at 1.0 s
    tone = make_flac(tmp_path / "a.flac", 3, freq=1000, on=[(1.0, 3.0)])
    eq = Step("eq", "firequalizer", {"delay": 0.05})
    graph = f"[0:a]{MONO},{chain_filter([eq], {})},silencedetect=n=-40dB:d=0.2"
    done = ffmpeg.run(
        ["-i", str(tone), "-filter_complex", graph, "-f", "null", "-"], "t"
    )
    start = re.search(r"silence_end: ([\d.]+)", done.stderr)
    assert start is not None and float(start.group(1)) == pytest.approx(1.0, abs=0.01)


def test_chain_leaves_no_denormals(tmp_path):
    # a short sound, then digital silence: the highpass tail decays into
    # denormals (very slow floats) unless adenorm follows it
    blip = make_flac(tmp_path / "a.flac", 3, on=[(0.5, 0.7)])
    hp = Step("rumble", "highpass", {"f": 80})
    stats = "astats=measure_perchannel=none:measure_overall=Number_of_denormals"
    graph = f"[0:a]{MONO},{chain_filter([hp], {})},{stats}"
    done = ffmpeg.run(
        ["-i", str(blip), "-filter_complex", graph, "-f", "null", "-"], "t"
    )
    assert "Number of denormals: 0.000000" in done.stderr


# --- mix and render ---------------------------------------------------------


def file_loudness(path):
    return measure(
        ["-i", str(path)], "[0:a]ebur128=dualmono=true:framelog=verbose", "t"
    )


def peak_db(path):
    import re

    from scribe_rpg import ffmpeg

    done = ffmpeg.run(
        ["-i", str(path), "-af", "astats=measure_perchannel=none", "-f", "null", "-"],
        "peak",
    )
    return float(re.findall(r"Peak level dB:\s*(-?[\d.]+)", done.stderr)[-1])


def plans_for(*paths):
    from scribe_rpg.craig import Track
    from scribe_rpg.master.mix import TrackPlan

    return [TrackPlan(Track(p.stem, "player", p), []) for p in paths]


def test_amix_sums_without_halving(tmp_path):
    from scribe_rpg.master.mix import render

    a = make_flac(tmp_path / "a.flac", 5, volume_db=-20)
    b = make_flac(tmp_path / "b.flac", 5, volume_db=-20)
    out = tmp_path / "sum.flac"
    render(plans_for(a, b), [], {}, "mono", out, "flac", "", None)
    assert file_loudness(out) - file_loudness(a) == pytest.approx(6.02, abs=0.2)


def test_limiter_caps_the_mix(tmp_path):
    from scribe_rpg.master.mix import render

    a = make_flac(tmp_path / "a.flac", 5, volume_db=-3)
    b = make_flac(tmp_path / "b.flac", 5, volume_db=-3)
    out = tmp_path / "sum.flac"
    limiter = [Step("limiter", "alimiter", {"limit": "-1dB", "level": False})]
    render(plans_for(a, b), limiter, {}, "mono", out, "flac", "", None)
    assert peak_db(out) <= -0.9


def test_render_reports_progress(tmp_path):
    from scribe_rpg.master.mix import render

    a = make_flac(tmp_path / "a.flac", 3)
    seen = []
    render(plans_for(a), [], {}, "mono", tmp_path / "o.flac", "flac", "", seen.append)
    assert seen and seen[-1] == pytest.approx(100, abs=1)


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_render_failure_leaves_no_file(tmp_path, monkeypatch, error):
    from scribe_rpg import ffmpeg
    from scribe_rpg.master import mix

    a = make_flac(tmp_path / "a.flac", 3)
    out = tmp_path / "o.mp3"

    def half_written(args, label, on_time=None):
        out.write_bytes(b"half")
        raise error()

    monkeypatch.setattr(ffmpeg, "run_progress", half_written)
    with pytest.raises(error):
        mix.render(plans_for(a), [], {}, "mono", out, "mp3", "96k", None)
    assert not out.exists()


# --- config check -------------------------------------------------------------


def checked(tmp_path, text):
    from scribe_rpg.config import load_config
    from scribe_rpg.master.config import resolve
    from scribe_rpg.master.mix import check

    path = tmp_path / "scribe.yaml"
    path.write_text("master:\n" + text, encoding="utf-8")
    check(resolve(load_config(path).settings["master"]))


def test_check_default_config_passes(tmp_path):
    checked(tmp_path, "  format: mp3\n")


def test_check_unknown_option(tmp_path):
    with pytest.raises(ScribeError) as e:
        checked(tmp_path, "  voice:\n    comp: {level: 3}\n")
    # ffmpeg 9 wording; older versions say "Option 'level' not found"
    assert str(e.value).startswith("master.voice.comp (acompressor): ")
    assert "'level'" in str(e.value) and "not found" in str(e.value)
    assert "|" not in str(e.value) and "@" not in str(e.value)


def test_check_value_out_of_range(tmp_path):
    with pytest.raises(ScribeError) as e:
        checked(tmp_path, "  voice:\n    comp: {ratio: 99}\n")
    assert str(e.value) == (
        "master.voice.comp (acompressor): "
        "Value 99.000000 for parameter 'ratio' out of range [1 - 20]"
    )


def test_check_reports_player_step(tmp_path):
    with pytest.raises(ScribeError) as e:
        checked(
            tmp_path,
            "  players:\n    player_one:\n      chain: [gate, loudness]\n"
            "      gate: {threshold: 7}\n",
        )
    assert str(e.value).startswith("master.players.player_one.gate (agate): ")
