"""The master: section: defaults, user tweaks, resolved chains."""

from pathlib import Path

from scribe_rpg.config import load_config
from scribe_rpg.craig import Track
from scribe_rpg.master.catalogue import Step
from scribe_rpg.master.config import BITRATES, resolve


def defaults():
    return resolve(load_config().settings["master"])


def test_default_chains():
    cfg = defaults()
    assert (cfg.channels, cfg.format, cfg.bitrate) == ("mono", "mp3", "96k")
    assert [(s.name, s.type) for s in cfg.voice] == [
        ("rumble", "highpass"),
        ("level", "loudness"),
        ("comp", "acompressor"),
        ("loudness", "loudness"),
    ]
    assert cfg.voice[0] == Step("rumble", "highpass", {"f": 80}, "master.voice.rumble")
    assert cfg.voice[2].options["threshold"] == "-15dB"
    assert cfg.music == [
        Step("loudness", "loudness", {"target": -36}, "master.music.loudness")
    ]
    assert [(s.name, s.type) for s in cfg.mix] == [("limiter", "alimiter")]
    assert cfg.mix[0].options["level"] is False
    assert cfg.players == {}


def test_bitrate_default_per_format():
    section = load_config().settings["master"]
    for fmt, rate in BITRATES.items():
        assert resolve({**section, "format": fmt}).bitrate == rate
    assert resolve({**section, "bitrate": "128k"}).bitrate == "128k"


def test_user_tweak_changes_one_option(tmp_path):
    path = tmp_path / "scribe.yaml"
    path.write_text("master:\n  voice:\n    comp: {ratio: 4}\n", encoding="utf-8")
    cfg = resolve(load_config(path).settings["master"])
    comp = defaults().voice[2]
    assert cfg.voice[2] == Step(
        "comp", "acompressor", {**comp.options, "ratio": 4}, "master.voice.comp"
    )
    assert cfg.voice[:2] == defaults().voice[:2]


def test_user_chain_may_name_any_default_step(tmp_path):
    path = tmp_path / "scribe.yaml"
    path.write_text(
        "master:\n  voice:\n    chain: [rumble, gate, loudness]\n", encoding="utf-8"
    )
    cfg = resolve(load_config(path).settings["master"])
    assert [s.name for s in cfg.voice] == ["rumble", "gate", "loudness"]


def test_chain_for():
    cfg = defaults()
    music = Track("Kenku_FM_1234", "music", Path("3-Kenku_FM_1234.flac"))
    voice = Track("gm_account", "player", Path("1-gm_account.flac"))
    assert cfg.chain_for(music) == cfg.music
    assert cfg.chain_for(voice) == cfg.voice


def test_player_chain_merged_over_voice():
    section = load_config().settings["master"]
    section = {
        **section,
        "players": {
            "Game Master": {"comp": {"threshold": "-20dB"}},
            "player.two": {"chain": ["rumble", "gate", "loudness"]},
        },
    }
    cfg = resolve(section)
    gm = cfg.chain_for(Track("Game Master", "player", Path("Game Master.flac")))
    assert gm[2].options["threshold"] == "-20dB"
    assert gm[2].where == "master.players.Game Master.comp"
    two = cfg.chain_for(Track("player_two", "player", Path("2-player_two.flac")))
    assert [s.name for s in two] == ["rumble", "gate", "loudness"]
