"""Tests for creating scribe.yaml and glossary.txt from the templates."""

import re
from importlib.resources import files

import pytest
import yaml

from scribe_rpg import init
from scribe_rpg.config import load_config
from scribe_rpg.errors import ScribeError
from scribe_rpg.master.config import resolve
from scribe_rpg.transcribe import session_json
from scribe_rpg.transcribe.glossary import load_glossary, parse_glossary

from helpers import craig_dir, craig_unzipped, craig_zip, raw_session

MUSIC = ["kenku"]


def test_templates_load_cleanly(tmp_path):
    init.write(tmp_path)
    cfg = load_config(tmp_path / "scribe.yaml")
    assert cfg.speakers["gm_account"] == "Game Master"
    assert cfg.glossary_path == tmp_path / "glossary.txt"
    gloss = load_glossary(cfg.glossary_path)
    # the examples are comments: a fresh glossary adds nothing to the prompt
    assert gloss.names == []
    assert gloss.correct("w Worgradzie") == "w Worgradzie"
    assert "Worgrad* -> Vorgrad" in init.template("glossary.txt")


def test_glossary_examples_work_once_uncommented():
    text = init.template("glossary.txt")
    # examples are commented out as "#Name" (no space after '#')
    lines = [ln[1:] if re.match(r"#\S", ln) else ln for ln in text.splitlines()]
    gloss = parse_glossary("\n".join(lines))
    assert gloss.names[:3] == ["Vorgrad", "Ostromir", "Czarna Wieża"]
    assert gloss.correct("w Worgradzie") == "w Vorgradzie"


def test_master_examples_work_once_uncommented(tmp_path):
    text = init.template("scribe.yaml")
    # the master: block's examples are commented out as "  # key" lines
    block = text[text.index("master:") : text.index("transcribe:")]
    uncommented = re.sub(r"(?m)^  # ", "  ", block)
    cfg = resolve(_load(tmp_path, text.replace(block, uncommented)).settings["master"])
    assert cfg.players_stereo["player_one"] is True
    assert [s.name for s in cfg.players["player_two"]] == [
        "rumble",
        "level",
        "comp",
        "eq",
        "denoise",
        "loudness",
    ]
    assert cfg.music[-1].options["target"] == -36


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


def test_speakers_from_unzipped_craig_folder(tmp_path):
    players, music = init.speakers_from(craig_unzipped(tmp_path / "s"), MUSIC)
    assert players == ["gm_account", "player_one"]
    assert music == ["Kenku_FM_1234"]


def test_speakers_from_track_folder_uses_file_names(tmp_path):
    players, _ = init.speakers_from(craig_dir(tmp_path / "tracks"), MUSIC)
    assert players == ["gm_account", "player_one"]


def test_speakers_from_folder_without_info_txt(tmp_path):
    names = ["1-gm_account.flac", "Game Master.flac", "3-Kenku_FM_1234.flac"]
    folder = craig_dir(tmp_path / "tracks", info=False, names=names)
    players, music = init.speakers_from(folder, MUSIC)
    assert players == ["gm_account", "Game Master"]
    assert music == ["Kenku_FM_1234"]


def test_speakers_from_json(tmp_path):
    path = tmp_path / "s.json"
    session_json.write(path, raw_session([]))
    players, music = init.speakers_from(path, MUSIC)
    assert players == ["gm_account", "player_one"]
    assert music == ["Kenku_FM_1234"]


def _load(tmp_path, text):
    path = tmp_path / "scribe.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def _keys(text):
    """Top-level keys of a YAML text, in order."""
    return [k for k in yaml.safe_load(text)]


def test_short_template_lists_only_the_usual_keys():
    text = init.render_config()
    assert _keys(text) == ["speakers", "input", "master", "transcribe"]
    assert "phrases" not in text and "--full" in text


def test_render_only_master(tmp_path):
    text = init.render_config(["gm.account", "player.one"], ["Kenku"], only="master")
    assert _keys(text) == ["input", "master"]
    assert '# accounts in this recording: "gm.account", "player.one"' in text
    cfg = _load(tmp_path, text)
    assert cfg.speakers == {} and cfg.glossary_path is None
    assert cfg.settings["master"]["format"] == "mp3"


def test_render_only_transcribe(tmp_path):
    text = init.render_config(["gm.account"], only="transcribe", language="pl")
    assert _keys(text) == ["speakers", "input", "transcribe"]
    cfg = _load(tmp_path, text)
    assert cfg.speakers == {"gm_account": "gm.account"}
    assert cfg.settings["transcribe"]["whisper"]["language"] == "pl"


def test_render_full_holds_every_default(tmp_path):
    text = init.render_config(full=True, language="pl")
    assert _keys(text) == ["speakers", "input", "master", "transcribe"]
    defaults_text = files("scribe_rpg").joinpath("defaults.yaml").read_text("utf-8")
    defaults = yaml.safe_load(defaults_text)
    data = yaml.safe_load(text)
    # steps in no chain are built in; a config defining one fails, so they
    # are written as comments
    voice = defaults["master"]["voice"]
    for name in [k for k in voice if k not in [*voice["chain"], "chain", "stereo"]]:
        del voice[name]
        assert f"    # {name}:" in text
    assert data["master"] == defaults["master"]
    assert data["input"] == defaults["input"]
    assert data["transcribe"].pop("glossary") == "glossary.txt"
    assert data["transcribe"]["whisper"].pop("language") == "pl"
    defaults["transcribe"]["whisper"].pop("language")
    assert data["transcribe"] == defaults["transcribe"]
    assert "every setting with its default" in text
    assert _load(tmp_path, text).glossary_path == tmp_path / "glossary.txt"


def test_render_full_only_master():
    text = init.render_config(full=True, only="master")
    assert _keys(text) == ["input", "master"]
    assert "rumble:" in text


def test_write_only_master_skips_glossary(tmp_path):
    written, kept = init.write(tmp_path, only="master")
    assert [p.name for p in written] == ["scribe.yaml"] and kept == []
    assert not (tmp_path / "glossary.txt").exists()
