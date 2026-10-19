"""Tests for the merge pipeline: raw session -> lines and report data."""

import pytest

from scribeio.config import load_config
from scribeio.transcribe import pipeline
from scribeio.transcribe.glossary import Glossary, parse_glossary

from helpers import raw_session, seg

NAMES = {"gm_account": "Game Master", "player_one": "Player One"}


def segments():
    return [
        seg("gm_account", 1.0, "Witajcie na sesji"),
        seg("gm_account", 15.0, "Dzięki za oglądanie!", voiced=False),
        seg("player_one", 20.0, "Yyy."),
        seg("player_one", 30.0, "Idziemy do Worgradu"),
    ]


def run(raw=None, names=NAMES, glossary=None, **cleanup):
    cfg = load_config().settings["transcribe"]
    cfg["cleanup"].update(cleanup)
    gloss = glossary or parse_glossary("Worgrad* -> Vorgrad")
    return pipeline.run(raw or raw_session(segments()), cfg, names, gloss)


def test_lines_after_cleanup_and_glossary():
    result = run()
    assert [(line.speaker, line.text) for line in result.lines] == [
        ("Game Master", "Witajcie na sesji"),
        ("Player One", "Idziemy do Vorgradu"),
    ]


def test_dropped_with_reasons():
    assert [d.reason for d in run().dropped] == ["energy", "filler"]


def test_keep_fillers():
    texts = [line.text for line in run(keep_fillers=True).lines]
    assert "Yyy." in texts


def test_pause_comes_from_lines_section():
    # gap 1.8 s: two lines with the default pause (1.5), one with pause 2.0
    raw = raw_session(
        [seg("gm_account", 0.0, "raz dwa"), seg("gm_account", 2.6, "trzy")]
    )
    cfg = load_config().settings["transcribe"]
    assert len(pipeline.run(raw, cfg, NAMES, Glossary()).lines) == 2
    cfg["lines"]["pause"] = 2.0
    result = pipeline.run(raw, cfg, NAMES, Glossary())
    assert [line.text for line in result.lines] == ["raz dwa trzy"]


def test_speakers_in_order_of_first_line():
    raw = raw_session(
        [seg("player_one", 1.0, "Zaczynam"), seg("gm_account", 5.0, "Dobrze")]
    )
    assert run(raw).speakers == ["Player One", "Game Master"]


def test_stats_per_track():
    stats = {s.account: s for s in run().stats}
    gm = stats["gm_account"]
    assert (gm.speaker, gm.kind, gm.kept, gm.dropped, gm.words) == (
        "Game Master",
        "player",
        1,
        1,
        3,
    )
    assert gm.speaking_time == pytest.approx(1.2)
    music = stats["Kenku_FM_1234"]
    assert (music.kind, music.kept, music.dropped) == ("music", 0, 0)


def test_stats_follow_track_order():
    assert [s.account for s in run().stats] == [
        "gm_account",
        "player_one",
        "Kenku_FM_1234",
    ]


def test_unknown_account_warning():
    result = run(names={"gm_account": "Game Master"})
    assert any("player_one" in w for w in result.warnings)
    assert result.speaker_of("player_one") == "player_one"


def test_prompt_cut_warning():
    whisper = {"initial_prompt": "Vorgrad", "prompt_cut": ["Ostromir"]}
    raw = raw_session(segments(), whisper=whisper)
    assert any("Ostromir" in w for w in run(raw).warnings)


def test_estimated_words_warning():
    raw = raw_session([seg("gm_account", 1.0, "rok 1923", estimated=True)])
    assert any(w.startswith("2 words") for w in run(raw).warnings)


def test_run_warnings_come_first():
    raw = raw_session(segments(), warnings=["runs on cpu: very slow"])
    assert run(raw).warnings == ["runs on cpu: very slow"]


def test_no_warnings_when_all_known():
    assert run().warnings == []


def test_bleed_reported_not_removed():
    text = "idziemy dalej korytarzem w stronę światła"
    raw = raw_session(
        [
            seg("gm_account", 100.0, text, level_db=-18.0),
            seg("player_one", 100.1, text, level_db=-41.0),
        ]
    )
    result = run(raw)
    assert len(result.bleed) == 1
    assert {line.speaker for line in result.lines} == {"Game Master", "Player One"}


def test_all_dropped():
    raw = raw_session([seg("gm_account", 1.0, "cisza tutaj", voiced=False)])
    result = run(raw, glossary=Glossary())
    assert result.lines == [] and result.speakers == []
    assert len(result.dropped) == 1


def test_raw_session_not_modified():
    raw = raw_session(segments())
    before = [s.text for s in raw.segments]
    run(raw)
    assert [s.text for s in raw.segments] == before
