"""Tests for the transcript (.md) and the quality report."""

from pathlib import Path

import yaml

from scribe_rpg.config import load_config
from scribe_rpg.transcribe import markdown, pipeline, report
from scribe_rpg.transcribe.glossary import Glossary
from scribe_rpg.transcribe.session import Segment, Word

from helpers import raw_session, seg

DATA = Path(__file__).parent / "data"
NAMES = {"gm_account": "Game Master", "player_one": "Player One"}


def merged(raw, names=NAMES):
    return pipeline.run(raw, load_config().settings["transcribe"], names, Glossary())


def spec_example():
    t0 = 3730.0
    texts = ["Wchodzicie", "do", "piwnicy,", "jest", "ciemno", "i", "czuć", "wilgoć..."]
    words = [
        Word(t, t0 + i * 0.5, t0 + (i + 1) * 0.5, 0.9) for i, t in enumerate(texts)
    ]
    gm = Segment("gm_account", t0, t0 + 4.0, " ".join(texts), words, -0.3, -20.0)
    tak = [Word("Tak.", t0 + 3.0, t0 + 3.4, 0.9)]
    p1 = Segment("player_one", t0 + 3.0, t0 + 3.4, "Tak.", tak, -0.3, -22.0)
    return raw_session([gm, p1])


def frontmatter(text):
    return yaml.safe_load(text.split("---\n")[1])


def test_hms_and_duration():
    assert markdown.hms(3733.9) == "01:02:13"
    assert markdown.hms(0) == "00:00:00"
    assert markdown.duration(13850.4) == "3:50:50"


def test_markdown_matches_expected_file(tmp_path):
    raw = spec_example()
    path = tmp_path / "out.md"
    markdown.write(path, raw, merged(raw))
    assert path.read_bytes() == (DATA / "expected.md").read_bytes()


def test_frontmatter_survives_special_characters():
    raw = raw_session(
        [seg("gm_account", 1.0, "Cześć")], source='sesja: "#1" [beta].zip'
    )
    text = markdown.render(raw, merged(raw, {"gm_account": 'GM: "Narrator" [#1]'}))
    fm = frontmatter(text)
    assert fm["speakers"] == ['GM: "Narrator" [#1]']
    assert fm["source"] == 'sesja: "#1" [beta].zip'
    assert fm["start"] == "19:00"


def test_empty_transcript():
    raw = raw_session([seg("gm_account", 1.0, "cisza", voiced=False)])
    text = markdown.render(raw, merged(raw))
    assert frontmatter(text)["speakers"] == []
    assert text.rstrip().endswith("---")
    assert "## Summary" in report.render(raw, merged(raw))


def report_sample():
    same = "idziemy dalej korytarzem w stronę światła"
    return raw_session(
        [
            seg("gm_account", 1.0, "Witajcie na sesji"),
            seg("gm_account", 15.0, "Dzięki za oglądanie!", voiced=False),
            seg("player_one", 20.0, "Yyy."),
            seg("gm_account", 100.0, same, level_db=-18.0),
            seg("player_one", 100.1, same, level_db=-41.0),
        ]
    )


def test_report_sections_in_order():
    raw = report_sample()
    text = report.render(raw, merged(raw))
    heads = [line for line in text.splitlines() if line.startswith("## ")]
    assert heads == [
        "## Summary",
        "## Warnings",
        "## Dropped segments",
        "## Possible bleed",
    ]


def test_report_content():
    raw = report_sample()
    text = report.render(raw, merged(raw))
    versions = (
        "scribe 0.1.0 · whisperx 3.8.6 · torch 2.8.0+cu128 · ffmpeg 7.1 · large-v3"
    )
    assert versions in text
    assert "| player_one | Player One | player |" in text
    assert "| Kenku_FM_1234 | Kenku_FM_1234 | music | 0 | 0 | 0 | 0:00:00 |" in text
    assert "- [00:00:15] Game Master: Dzięki za oglądanie! (0% voiced)" in text
    assert "- [00:00:20] Player One: Yyy." in text
    assert (
        "- [00:01:40] Game Master (-18 dB) / Player One (-41 dB): "
        "idziemy dalej korytarzem w stronę światła"
    ) in text
    assert "None." in text  # no warnings


def test_report_lists_warnings():
    raw = report_sample()
    text = report.render(raw, merged(raw, {"gm_account": "Game Master"}))
    expected = "- unknown accounts (no speakers: entry, account name used): player_one"
    assert expected in text


def test_report_without_bleed_or_drops():
    raw = raw_session([seg("gm_account", 1.0, "Witajcie na sesji")])
    text = report.render(raw, merged(raw))
    assert "### Energy (track silent under most words) (0)" in text
    assert text.rstrip().endswith("None.")


def test_report_bleed_level_unknown():
    same = "idziemy dalej korytarzem w stronę światła"
    raw = raw_session(
        [
            seg("gm_account", 100.0, same, level_db=None),
            seg("player_one", 100.1, same, level_db=-41.0),
        ]
    )
    assert "Game Master (no level) / Player One (-41 dB)" in report.render(
        raw, merged(raw)
    )


def test_write_uses_utf8_and_lf(tmp_path):
    raw = report_sample()
    path = tmp_path / "r.md"
    report.write(path, raw, merged(raw))
    data = path.read_bytes()
    assert b"\r\n" not in data and "światła".encode() in data
