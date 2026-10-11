"""The example config and glossary shipped in examples/ must load cleanly."""

from pathlib import Path

from scribe_rpg.config import load_config
from scribe_rpg.transcribe.glossary import load_glossary

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_example_config_loads():
    cfg = load_config(EXAMPLES / "campaign.yaml")
    assert cfg.speakers["gm_account"] == "Game Master"
    assert cfg.speakers["player_two_old"] == "Player Two"
    assert cfg.settings["input"]["music_patterns"] == ["kenku", "jukebox"]
    assert cfg.settings["transcribe"]["lines"]["pause"] == 1.5
    assert cfg.glossary_path == EXAMPLES / "glossary.txt"


def test_example_glossary_loads_and_corrects():
    gloss = load_glossary(EXAMPLES / "glossary.txt")
    assert gloss.names[:3] == ["Vorgrad", "Ostromir", "Czarna Wieża"]
    assert gloss.correct("w Worgradzie") == "w Vorgradzie"
    assert gloss.correct("przy czarnej wierzy") == "przy Czarnej Wieży"
