"""Filter text built from steps: quoting, loudness steps, measurement."""

from pathlib import Path

import pytest

from scribeio import ffmpeg
from scribeio.master.catalogue import FILTERS, Step, quote
from scribeio.master.mix import discard, partial_path
from scribeio.master.render import (
    chain_filter,
    measure_filter,
    parse_integrated,
    parse_peak,
    step_filter,
)

LOUD = Step("level", "loudness", {"target": -23})


@pytest.mark.parametrize("kind", sorted(FILTERS))
def test_every_filter_type(kind):
    assert step_filter(Step("s", kind, {})) == kind
    assert step_filter(Step("s", kind, {"a": 1, "b": "x"})) == f"{kind}=a=1:b=x"


def test_values_pass_unchanged():
    step = Step("comp", "acompressor", {"threshold": "-15dB", "ratio": 2.5})
    assert step_filter(step) == "acompressor=threshold=-15dB:ratio=2.5"


def test_bool_values():
    assert quote(True) == "true" and quote(False) == "false"


@pytest.mark.parametrize(
    ("value", "quoted"),
    [
        ("a:b", r"a\\:b"),
        ("a,b", r"a\,b"),
        ("it's", r"it\\\'s"),
        ("a;b", r"a\;b"),
        ("[x]", r"\[x\]"),
        ("c:\\d", r"c\\:\\\\d"),
        ("entry(100,0);entry(200,3)", r"entry(100\,0)\;entry(200\,3)"),
    ],
)
def test_quote(value, quoted):
    assert quote(value) == quoted


def test_loudness_with_and_without_gain():
    assert step_filter(LOUD, 4.256) == "volume=4.26dB"
    assert step_filter(LOUD, -3.0) == "volume=-3.00dB"
    assert step_filter(LOUD, None) == "anull"


def test_chain_filter():
    steps = [Step("hp", "highpass", {"f": 80}), LOUD, Step("lim", "alimiter", {})]
    assert chain_filter(steps, {1: 2.0}) == (
        "aformat=sample_fmts=flt,highpass=f=80,adenorm,volume=2.00dB,alimiter,adenorm"
    )
    assert chain_filter(steps, {}, upto=1) == (
        "aformat=sample_fmts=flt,highpass=f=80,adenorm"
    )
    assert chain_filter([], {}) == "aformat=sample_fmts=flt"


def test_chain_filter_missing_gain():
    with pytest.raises(ValueError):
        chain_filter([LOUD], {})


def test_firequalizer_delay_is_compensated():
    trim = "atrim=start={},asetpts=PTS-STARTPTS"
    eq = Step("eq", "firequalizer", {"delay": 0.05})
    assert chain_filter([eq], {}) == (
        f"aformat=sample_fmts=flt,firequalizer=delay=0.05,adenorm,{trim.format(0.05)}"
    )
    # ffmpeg's default delay is 0.01 s
    plain = Step("eq", "firequalizer", {})
    assert chain_filter([plain], {}).endswith(f",adenorm,{trim.format(0.01)}")


def test_measure_filter_dual_mono():
    steps = [Step("hp", "highpass", {"f": 80}), LOUD]
    assert measure_filter(steps, {}, 1) == (
        "aformat=sample_fmts=flt,highpass=f=80,adenorm,"
        "ebur128=dualmono=true:framelog=verbose"
    )
    mono = Step("level", "loudness", {"target": -23, "dual_mono": False})
    assert measure_filter([mono], {}, 0).endswith(
        ",ebur128=dualmono=false:framelog=verbose"
    )


@pytest.mark.ffmpeg
def test_quoted_text_is_accepted_by_ffmpeg():
    steps = [
        Step("eq", "firequalizer", {"gain_entry": "entry(100,0);entry(200,3)"}),
        Step("v", "volume", {"volume": "-3dB"}),
    ]
    ffmpeg.run(
        ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "0.1",
         "-af", chain_filter(steps, {}), "-f", "null", "-"],
        "quoting",
    )  # fmt: skip


@pytest.mark.ffmpeg
def test_quoted_text_reaches_the_filter_unchanged():
    # Both unescape levels: a lone ":" would start a second option.
    value = "a:b=1,c;d'e\\f[g]"
    steps = [
        Step("tag", "ametadata", {"mode": "add", "key": "k", "value": value}),
        Step("show", "ametadata", {"mode": "print", "key": "k"}),
    ]
    done = ffmpeg.run(
        ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "0.05",
         "-af", chain_filter(steps, {}), "-f", "null", "-"],
        "quoting",
    )  # fmt: skip
    assert f"] k={value}\n" in done.stderr


# Captured from ffmpeg 7.1; the per-frame line's "I:" must not be read.
SUMMARY = """\
[Parsed_ebur128_1 @ 0] t: 2.99  M: -18.1 S: -18.1   I: -18.1 LUFS   LRA:  20.0 LU
[Parsed_ebur128_1 @ 0] Summary:

  Integrated loudness:
    I:         -21.4 LUFS
    Threshold: -31.4 LUFS

  Loudness range:
    LRA:        20.0 LU
"""


def test_parse_integrated_reads_the_summary():
    assert parse_integrated(SUMMARY) == -21.4


@pytest.mark.parametrize("value", ["-70.0", "-inf", "-91.3"])
def test_parse_integrated_silence(value):
    assert parse_integrated(SUMMARY.replace("-21.4", value)) is None


def test_parse_integrated_without_summary():
    assert parse_integrated("no ebur128 here") is None


PEAK = (
    SUMMARY
    + """
  Sample peak:
    Peak:       -1.0 dBFS
"""
)


def test_parse_peak_reads_the_summary():
    assert parse_peak(PEAK) == -1.0


def test_parse_peak_silence_or_absent():
    assert parse_peak(PEAK.replace("-1.0 dBFS", "-inf dBFS")) is None
    assert parse_peak("no ebur128 here") is None


def test_mix_graph_two_voices_and_music_mono():
    from pathlib import Path

    from scribeio.craig import Track
    from scribeio.master.mix import TrackPlan, graph

    plans = [
        TrackPlan(
            Track("a", "player", Path("1-a.flac")), [Step("hp", "highpass", {"f": 80})]
        ),
        TrackPlan(Track("b", "player", Path("2-b.flac")), []),
        TrackPlan(Track("m", "music", Path("3-m.flac")), [LOUD], {0: -3.0}),
    ]
    limiter = [Step("limiter", "alimiter", {"limit": "-1dB"})]
    mono = "aformat=sample_fmts=flt:channel_layouts=mono,aformat=sample_fmts=flt"
    assert graph(plans, limiter, {}, "mono") == (
        f"[0:a]{mono},highpass=f=80,adenorm,apad=pad_dur=5[t0];"
        f"[1:a]{mono},apad=pad_dur=5[t1];"
        f"[2:a]{mono},volume=-3.00dB,apad=pad_dur=5[t2];"
        "[t0][t1][t2]amix=inputs=3:normalize=0:duration=longest,"
        "aformat=sample_fmts=flt,alimiter=limit=-1dB,adenorm[out]"
    )


def test_prep_mono_and_stereo():
    from pathlib import Path

    from scribeio.craig import Track
    from scribeio.master.mix import TrackPlan, prep

    voice = TrackPlan(Track("a", "player", Path("1-a.flac")), [])
    stereo_voice = TrackPlan(Track("b", "player", Path("2-b.flac")), [], stereo=True)
    mono = "aformat=sample_fmts=flt:channel_layouts=mono"
    stereo = "aformat=sample_fmts=flt:channel_layouts=stereo"
    # mono output: everything read as mono
    assert prep(voice, "mono") == mono and prep(stereo_voice, "mono") == mono
    # stereo output: a track is read as stereo only if it keeps its stereo
    assert prep(voice, "stereo") == mono
    assert prep(stereo_voice, "stereo") == stereo


def test_mix_graph_stereo_pans_mono_voices_after_their_chain():
    from pathlib import Path

    from scribeio.craig import Track
    from scribeio.master.mix import TrackPlan, graph

    hp = [Step("hp", "highpass", {"f": 80})]
    plans = [
        TrackPlan(Track("a", "player", Path("1-a.flac")), hp),
        TrackPlan(Track("b", "player", Path("2-b.flac")), hp, stereo=True),
    ]
    mono = "aformat=sample_fmts=flt:channel_layouts=mono,aformat=sample_fmts=flt"
    stereo = "aformat=sample_fmts=flt:channel_layouts=stereo,aformat=sample_fmts=flt"
    assert graph(plans, [], {}, "stereo") == (
        f"[0:a]{mono},highpass=f=80,adenorm,pan=stereo|c0=c0|c1=c0,apad=pad_dur=5[t0];"
        f"[1:a]{stereo},highpass=f=80,adenorm,apad=pad_dur=5[t1];"
        "[t0][t1]amix=inputs=2:normalize=0:duration=longest,"
        "aformat=sample_fmts=flt[out]"
    )


def test_discard_retries_a_held_file(tmp_path, monkeypatch):
    # a killed ffmpeg shim's child may hold the file for a moment
    path = tmp_path / "x.partial.mp3"
    path.write_bytes(b"half")
    real = Path.unlink
    tries = []

    def held(self, missing_ok=False):
        tries.append(self)
        if len(tries) < 3:
            raise PermissionError("in use")
        real(self, missing_ok)

    monkeypatch.setattr(Path, "unlink", held)
    discard(path)
    assert not path.exists() and len(tries) == 3


def test_discard_gives_up_without_raising(tmp_path, monkeypatch):
    folder = tmp_path / "s1_tracks.partial"
    folder.mkdir()

    def held(path, *args, **kwargs):
        raise PermissionError("in use")

    monkeypatch.setattr("shutil.rmtree", held)
    discard(folder, wait=0.2)  # no exception: the real error is what matters
    assert folder.exists()
    discard(tmp_path / "missing")  # nothing to do


def test_partial_path_keeps_a_folder_name_whole():
    # files: the marker before the suffix, so players still open them
    assert partial_path(Path("s1.v2_master.mp3")) == Path("s1.v2_master.partial.mp3")
    folder = Path("s1.v2_tracks")
    assert partial_path(folder, folder=True) == Path("s1.v2_tracks.partial")
