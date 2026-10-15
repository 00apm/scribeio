"""Tests for writing and reading the raw session JSON."""

import json

import pytest

from scribe_rpg.errors import ScribeError
from scribe_rpg.transcribe import session_json

from helpers import raw_session, seg


def sample():
    return raw_session(
        [
            seg("gm_account", 1.0, "Witajcie w Vorgradzie", level_db=-21.5),
            seg(
                "player_one", 3.25, "Tak.", voiced=False, level_db=None, estimated=True
            ),
        ]
    )


def test_round_trip(tmp_path):
    path = tmp_path / "s.json"
    session_json.write(path, sample())
    assert session_json.read(path) == sample()


def test_layout_and_utf8(tmp_path):
    path = tmp_path / "s.json"
    session_json.write(path, sample())
    text = path.read_text(encoding="utf-8")
    assert "Witajcie w Vorgradzie" in text and "\\u" not in text  # ensure_ascii=False
    data = json.loads(text)
    assert data["format"] == 1
    assert list(data) == [
        "format",
        "scribe",
        "versions",
        "session",
        "whisper",
        "tracks",
        "segments",
    ]
    assert data["session"] == {
        "date": "2026-01-01",
        "start": "2026-01-01T19:00:00+01:00",
        "duration": 3600.0,
        "source": "session.zip",
    }
    assert data["versions"]["whisperx"] == "3.8.6"
    segment = data["segments"][0]
    assert list(segment) == [
        "account",
        "start",
        "end",
        "text",
        "avg_logprob",
        "level_db",
        "words",
    ]
    assert list(segment["words"][0]) == [
        "word",
        "start",
        "end",
        "score",
        "voiced",
        "estimated",
    ]


def test_times_rounded_to_ms(tmp_path):
    raw = sample()
    raw.segments[0].start = 1.23456789
    raw.segments[0].words[0].end = 1.98765
    path = tmp_path / "s.json"
    session_json.write(path, raw)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["segments"][0]["start"] == 1.235
    assert data["segments"][0]["words"][0]["end"] == 1.988


def test_written_with_lf_line_endings(tmp_path):
    path = tmp_path / "s.json"
    session_json.write(path, sample())
    assert b"\r\n" not in path.read_bytes()


def test_file_without_versions_still_reads(tmp_path):
    path = tmp_path / "s.json"
    session_json.write(path, sample())
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["versions"]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert session_json.read(path).versions == {}


def test_unknown_format(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"format": 2}', encoding="utf-8")
    with pytest.raises(ScribeError, match="unknown format"):
        session_json.read(path)


def test_not_an_object(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ScribeError, match="unknown format"):
        session_json.read(path)


def test_not_json(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ScribeError, match="not valid JSON"):
        session_json.read(path)


def test_damaged(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"format": 1, "session": {}}', encoding="utf-8")
    with pytest.raises(ScribeError, match="damaged"):
        session_json.read(path)


def test_missing_file(tmp_path):
    with pytest.raises(ScribeError, match="not found"):
        session_json.read(tmp_path / "missing.json")
