"""Filter text built from steps: quoting, loudness steps, measurement."""

import pytest

from scribe_rpg import ffmpeg
from scribe_rpg.master.catalogue import FILTERS, Step, quote
from scribe_rpg.master.render import (
    chain_filter,
    measure_filter,
    parse_integrated,
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
        "aformat=sample_fmts=flt,highpass=f=80,volume=2.00dB,alimiter"
    )
    assert chain_filter(steps, {}, upto=1) == "aformat=sample_fmts=flt,highpass=f=80"
    assert chain_filter([], {}) == "aformat=sample_fmts=flt"


def test_chain_filter_missing_gain():
    with pytest.raises(ValueError):
        chain_filter([LOUD], {})


def test_measure_filter_dual_mono():
    steps = [Step("hp", "highpass", {"f": 80}), LOUD]
    assert measure_filter(steps, {}, 1) == (
        "aformat=sample_fmts=flt,highpass=f=80,ebur128=dualmono=true:framelog=verbose"
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


def test_mix_graph_two_voices_and_music_mono():
    from pathlib import Path

    from scribe_rpg.craig import Track
    from scribe_rpg.master.mix import TrackPlan, graph

    plans = [
        TrackPlan(
            Track("a", "player", Path("1-a.flac")), [Step("hp", "highpass", {"f": 80})]
        ),
        TrackPlan(Track("b", "player", Path("2-b.flac")), []),
        TrackPlan(Track("m", "music", Path("3-m.flac")), [LOUD], {0: -3.0}),
    ]
    limiter = [Step("limiter", "alimiter", {"limit": "-1dB"})]
    mono = "aformat=channel_layouts=mono,aformat=sample_fmts=flt"
    assert graph(plans, limiter, {}, "mono") == (
        f"[0:a]{mono},highpass=f=80[t0];"
        f"[1:a]{mono}[t1];"
        f"[2:a]{mono},volume=-3.00dB[t2];"
        "[t0][t1][t2]amix=inputs=3:normalize=0:duration=longest,"
        "aformat=sample_fmts=flt,alimiter=limit=-1dB[out]"
    )
