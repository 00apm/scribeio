"""Tests for creating scribe.yaml and glossary.txt from the templates."""

import pytest

from scribe_rpg import init
from scribe_rpg.config import load_config
from scribe_rpg.errors import ScribeError
from scribe_rpg.transcribe import session_json
from scribe_rpg.transcribe.glossary import load_glossary

from helpers import craig_dir, craig_zip, raw_session

MUSIC = ["kenku"]


def test_templates_load_cleanly(tmp_path):
    init.write(tmp_path)
    cfg = load_config(tmp_path / "scribe.yaml")
    assert cfg.speakers["gm_account"] == "Game Master"
    assert cfg.glossary_path == tmp_path / "glossary.txt"
    gloss = load_glossary(cfg.glossary_path)
    assert gloss.names[:3] == ["Vorgrad", "Ostromir", "Czarna Wieża"]
    assert gloss.correct("w Worgradzie") == "w Vorgradzie"


def test_render_without_changes_is_the_template():
    assert init.render_config() == init.template("scribe.yaml")


def test_render_prefilled_speakers_and_music(tmp_path):
    text = init.render_config(["gm.account", "odd: name#1234"], ["Kenku FM#1234"])
    path = tmp_path / "scribe.yaml"
    path.write_text(text, encoding="utf-8")
    speakers = load_config(path).speakers
    assert speakers == {"gm_account": "gm.account", "odd__name_1234": "odd: name#1234"}
    assert "# music bots (not transcribed): Kenku FM#1234" in text
    assert "Game Master" not in text  # template placeholders replaced


def test_render_language(tmp_path):
    path = tmp_path / "scribe.yaml"
    path.write_text(init.render_config(language="pl"), encoding="utf-8")
    assert load_config(path).settings["transcribe"]["whisper"]["language"] == "pl"


def test_write_refuses_to_overwrite_config(tmp_path):
    init.write(tmp_path)
    with pytest.raises(ScribeError, match="scribe.yaml already exists.*--force"):
        init.write(tmp_path)


def test_write_force_overwrites(tmp_path):
    init.write(tmp_path)
    (tmp_path / "scribe.yaml").write_text("OLD-CONTENT", encoding="utf-8")
    written, kept = init.write(tmp_path, force=True)
    assert "OLD-CONTENT" not in (tmp_path / "scribe.yaml").read_text(encoding="utf-8")
    assert kept == []


def test_write_keeps_existing_glossary(tmp_path):
    (tmp_path / "glossary.txt").write_text("Mine\n", encoding="utf-8")
    written, kept = init.write(tmp_path)
    assert [p.name for p in written] == ["scribe.yaml"]
    assert [p.name for p in kept] == ["glossary.txt"]
    assert (tmp_path / "glossary.txt").read_text(encoding="utf-8") == "Mine\n"


def test_write_creates_folder(tmp_path):
    written, _ = init.write(tmp_path / "new" / "folder")
    assert all(p.exists() for p in written)


def test_find_config(tmp_path):
    assert init.find_config(tmp_path) is None
    init.write(tmp_path)
    assert init.find_config(tmp_path) == tmp_path / "scribe.yaml"


def test_speakers_from_zip(tmp_path):
    players, music = init.speakers_from(craig_zip(tmp_path), MUSIC)
    assert players == ["gm.account", "player.one"]
    assert music == ["Kenku FM#1234"]


def test_speakers_from_track_folder(tmp_path):
    players, _ = init.speakers_from(craig_dir(tmp_path / "tracks"), MUSIC)
    assert players == ["gm.account", "player.one"]


def test_speakers_from_json(tmp_path):
    path = tmp_path / "s.json"
    session_json.write(path, raw_session([]))
    players, music = init.speakers_from(path, MUSIC)
    assert players == ["gm_account", "player_one"]
    assert music == ["Kenku_FM_1234"]
