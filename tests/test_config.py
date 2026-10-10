"""Tests for loading the built-in defaults and a transcription config."""

import textwrap
from pathlib import Path

import pytest

from scribe_rpg.config import load_config
from scribe_rpg.errors import ScribeError


def write(folder: Path, text: str, name: str = "transcription.yaml") -> Path:
    path = folder / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_defaults_only():
    cfg = load_config()
    assert cfg.settings["whisper"]["model"] == "large-v3"
    assert cfg.settings["whisper"]["language"] == "en"
    assert cfg.settings["filters"]["pause"] == 1.5
    assert cfg.settings["music_patterns"] == ["kenku"]
    assert cfg.speakers == {}
    assert cfg.glossary_path is None


def test_defaults_cover_english_and_polish_phrases():
    phrases = load_config().settings["filters"]["phrases"]
    assert "thanks for watching" in phrases
    assert "dzięki za oglądanie" in phrases


def test_partial_override_merges_recursively(tmp_path):
    cfg = load_config(write(tmp_path, "whisper:\n  batch_size: 2\n"))
    assert cfg.settings["whisper"]["batch_size"] == 2
    assert cfg.settings["whisper"]["model"] == "large-v3"


def test_list_replaces_not_appends(tmp_path):
    cfg = load_config(write(tmp_path, "music_patterns: [kenku, jukebox]\n"))
    assert cfg.settings["music_patterns"] == ["kenku", "jukebox"]


def test_unknown_key(tmp_path):
    with pytest.raises(
        ScribeError, match=r"transcription\.yaml: whisper\.bach_size: unknown key"
    ):
        load_config(write(tmp_path, "whisper:\n  bach_size: 2\n"))


def test_unknown_top_level_key(tmp_path):
    with pytest.raises(ScribeError, match=r"transcription\.yaml: speaker: unknown key"):
        load_config(write(tmp_path, "speaker:\n  gm.account: Game Master\n"))


def test_wrong_type(tmp_path):
    with pytest.raises(ScribeError, match=r"filters\.pause"):
        load_config(write(tmp_path, "filters:\n  pause: long\n"))


def test_section_must_be_mapping(tmp_path):
    with pytest.raises(ScribeError, match=r"whisper: expected key: value pairs"):
        load_config(write(tmp_path, "whisper: large-v3\n"))


def test_int_accepted_for_float(tmp_path):
    cfg = load_config(write(tmp_path, "filters:\n  pause: 2\n"))
    assert cfg.settings["filters"]["pause"] == 2


def test_bool_not_accepted_for_number(tmp_path):
    with pytest.raises(ScribeError, match=r"filters\.pause"):
        load_config(write(tmp_path, "filters:\n  pause: true\n"))


def test_yaml_error_names_file_and_line(tmp_path):
    path = write(tmp_path, "whisper:\n  batch_size: 2\n  model: [unclosed\n")
    with pytest.raises(ScribeError, match=r"transcription\.yaml:\d+"):
        load_config(path)


def test_config_not_found(tmp_path):
    with pytest.raises(ScribeError, match="config not found"):
        load_config(tmp_path / "missing.yaml")


def test_speakers_sanitized(tmp_path):
    cfg = load_config(
        write(
            tmp_path,
            """\
            speakers:
              gm.account: Game Master
              player.two.old: Player Two
              żółw.gm: Player One
            """,
        )
    )
    assert cfg.speakers == {
        "gm_account": "Game Master",
        "player_two_old": "Player Two",
        "___w_gm": "Player One",
    }


def test_speaker_name_must_be_text(tmp_path):
    with pytest.raises(ScribeError, match="speakers"):
        load_config(write(tmp_path, "speakers:\n  gm.account:\n"))


def test_glossary_relative_to_config(tmp_path):
    sub = tmp_path / "campaign"
    sub.mkdir()
    cfg = load_config(write(sub, "glossary: names.txt\n"))
    assert cfg.glossary_path == sub / "names.txt"


def test_glossary_option_wins(tmp_path):
    other = tmp_path / "other.txt"
    cfg = load_config(write(tmp_path, "glossary: names.txt\n"), glossary=other)
    assert cfg.glossary_path == other


def test_keep_fillers_flag():
    assert load_config(keep_fillers=True).settings["filters"]["keep_fillers"] is True


def test_each_load_gets_fresh_defaults():
    load_config(keep_fillers=True)
    assert load_config().settings["filters"]["keep_fillers"] is False


def test_empty_config_file(tmp_path):
    assert load_config(write(tmp_path, "")).settings["filters"]["pause"] == 1.5


def test_top_level_must_be_mapping(tmp_path):
    with pytest.raises(ScribeError, match="top level"):
        load_config(write(tmp_path, "- just\n- a list\n"))
