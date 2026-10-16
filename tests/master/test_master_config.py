"""The master: section: defaults, user tweaks, resolved chains."""

import re
from pathlib import Path

import pytest

from scribe_rpg.config import load_config
from scribe_rpg.craig import Track
from scribe_rpg.errors import ScribeError
from scribe_rpg.master.catalogue import Step
from scribe_rpg.master.config import BITRATES, merge, resolve, unknown_players


def defaults():
    return resolve(load_config().settings["master"])


def test_default_chains():
    cfg = defaults()
    assert (cfg.channels, cfg.format, cfg.bitrate) == ("mono", "mp3", "96k")
    assert [(s.name, s.type) for s in cfg.voice] == [
        ("rumble", "highpass"),
        ("level", "loudness"),
        ("comp", "acompressor"),
        ("eq", "firequalizer"),
        ("loudness", "loudness"),
    ]
    assert cfg.voice[3].options == {
        "gain_entry": "entry(80,0);entry(150,1.5);entry(300,0);entry(2500,0);"
        "entry(3300,3);entry(5000,3);entry(6000,0)",
        "delay": 0.05,
    }
    assert cfg.voice[0] == Step("rumble", "highpass", {"f": 80}, "master.voice.rumble")
    assert cfg.voice[2].options["threshold"] == "-15dB"
    assert cfg.music == [
        Step("loudness", "loudness", {"target": -36}, "master.music.loudness")
    ]
    assert [(s.name, s.type) for s in cfg.mix] == [("limiter", "alimiter")]
    assert cfg.mix[0].options == {"limit": "-1.5dB", "level": False}
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


def load(tmp_path, text):
    path = tmp_path / "scribe.yaml"
    path.write_text("master:\n" + text, encoding="utf-8")
    return resolve(load_config(path).settings["master"])


def player(cfg, account):
    return cfg.chain_for(Track(account, "player", Path(f"1-{account}.flac")))


def test_player_chain_merged_over_voice(tmp_path):
    cfg = load(
        tmp_path,
        """\
  players:
    Game Master:
      comp: {threshold: -20dB}
    player.two:
      chain: [rumble, gate, level, comp, loudness]
      gate: {type: agate, threshold: -45dB}
""",
    )
    gm = player(cfg, "Game Master")
    assert gm[2].options["threshold"] == "-20dB"
    assert gm[2].options["ratio"] == 3  # rest of voice's comp kept
    assert gm[2].where == "master.players.Game Master.comp"
    two = player(cfg, "player_two")
    assert [s.name for s in two] == ["rumble", "gate", "level", "comp", "loudness"]
    assert two[1].options == {"threshold": "-45dB", "range": "-12dB"}
    assert player(cfg, "someone") == cfg.voice


def test_player_sees_user_voice_tweaks(tmp_path):
    cfg = load(
        tmp_path,
        "  voice:\n    comp: {ratio: 5}\n  players:\n    a:\n      comp: {attack: 5}\n",
    )
    assert player(cfg, "a")[2].options["ratio"] == 5
    assert player(cfg, "a")[2].options["attack"] == 5
    assert cfg.voice[2].options["attack"] == 15


def test_lists_replace_and_new_steps(tmp_path):
    cfg = load(
        tmp_path,
        """\
  voice:
    chain: [rumble, hiss, loudness]
    hiss: {type: lowpass, f: 7000}
  mix:
    chain: [limiter]
    limiter: {type: alimiter, limit: -2dB}
""",
    )
    assert [s.name for s in cfg.voice] == ["rumble", "hiss", "loudness"]
    assert cfg.voice[1] == Step("hiss", "lowpass", {"f": 7000}, "master.voice.hiss")
    assert cfg.mix[0].options == {"limit": "-2dB", "level": False}


def test_player_chain_names_default_step(tmp_path):
    cfg = load(tmp_path, "  players:\n    a:\n      chain: [denoise, loudness]\n")
    assert [s.type for s in player(cfg, "a")] == ["afftdn", "loudness"]


def test_top_values(tmp_path):
    cfg = load(tmp_path, "  channels: stereo\n  format: opus\n  bitrate: 64k\n")
    assert (cfg.channels, cfg.format, cfg.bitrate) == ("stereo", "opus", "64k")


def test_merge_without_user_section():
    section = load_config().settings["master"]
    assert merge(section, None, "x.yaml") == section


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("  colour: red\n", "master.colour: unknown key"),
        ("  channels: surround\n", "master.channels: expected mono or stereo"),
        ("  format: wav\n", "master.format: expected mp3, m4a or opus"),
        ("  bitrate: 96\n", "master.bitrate: expected e.g. 96k"),
        ("  bitrate: 96kbps\n", "master.bitrate: expected e.g. 96k"),
        ("  voice: [a]\n", "master.voice: expected key: value pairs"),
        ("  voice:\n    chain: rumble\n", "master.voice.chain: expected a list"),
        ("  voice:\n    chain: [rumble, 3]\n", "master.voice.chain: expected a list"),
        (
            "  voice:\n    chain: {type: volume}\n",
            "master.voice.chain: 'chain' is reserved",
        ),
        (
            "  voice:\n    chain: [rumble, gaet]\n",
            "master.voice.chain: unknown step 'gaet'",
        ),
        (
            "  players:\n    player.two:\n      chain: [rumble, gaet]\n",
            "master.players.player.two.chain: unknown step 'gaet'",
        ),
        (
            "  voice:\n    hiss: {f: 7000}\n",
            "master.voice.hiss: a new step needs a type",
        ),
        (
            "  voice:\n    hiss: {type: aecho}\n",
            "master.voice.hiss: unknown type 'aecho'",
        ),
        (
            "  voice:\n    comp: {type: agate}\n",
            "master.voice.comp: a tweak cannot change `type`; use a new step name",
        ),
        (
            "  players:\n    a:\n      comp: {type: agate}\n",
            "master.players.a.comp: a tweak cannot change `type`",
        ),
        ("  voice:\n    comp: 4\n", "master.voice.comp: expected key: value pairs"),
        (
            "  voice:\n    eq: {dumpfile: out.txt}\n",
            "master.voice.eq: option 'dumpfile' is not allowed",
        ),
        (
            # an option name carrying filter syntax would add filters
            '  voice:\n    comp:\n      "ratio=2,firequalizer=dumpfile=x:fixed": 0\n',
            "master.voice.comp: bad option name "
            "'ratio=2,firequalizer=dumpfile=x:fixed'",
        ),
        (
            "  voice:\n    comp: {Ratio: 2}\n",
            "master.voice.comp: bad option name 'Ratio'",
        ),
        (
            "  voice:\n    comp: {3: 2}\n",
            "master.voice.comp: bad option name '3'",
        ),
        (
            "  players:\n    a:\n      comp: {'ratio;x': 2}\n",
            "master.players.a.comp: bad option name 'ratio;x'",
        ),
        (
            "  voice:\n    hiss: {type: lowpass}\n",
            "master.voice.hiss: step is in no chain",
        ),
        (
            "  music:\n    hiss: {type: lowpass}\n",
            "master.music.hiss: step is in no chain",
        ),
        (
            "  players:\n    a:\n      hiss: {type: lowpass}\n",
            "master.players.a.hiss: step is in no chain",
        ),
        (
            "  voice:\n    chain: [boost]\n    boost: {type: loudness}\n",
            "master.voice.boost: loudness needs a target in LUFS, e.g. -19",
        ),
        (
            "  music:\n    loudness: {target: loud}\n",
            "master.music.loudness: loudness needs a target in LUFS, e.g. -19",
        ),
        (
            "  voice:\n    stereo: yes please\n",
            "master.voice.stereo: expected true or false",
        ),
        (
            "  music:\n    stereo: true\n",
            "master.music.stereo: only voice and players have `stereo`",
        ),
        (
            "  voice:\n    chain: [stereo]\n",
            "master.voice.chain: unknown step 'stereo'",
        ),
        (
            "  voice:\n    level: {targt: -30}\n",
            "master.voice.level: loudness has only target and dual_mono, not 'targt'",
        ),
        (
            "  voice:\n    level: {dual_mono: maybe}\n",
            "master.voice.level: dual_mono: expected true or false",
        ),
        (
            "  music:\n    loudness: {target: .nan}\n",
            "master.music.loudness: loudness needs a target in LUFS, e.g. -19",
        ),
        (
            "  voice:\n    chain: [x]\n    x: {type: {a: 1}}\n",
            "master.voice.x: unknown type {'a': 1}",
        ),
        (
            "  players:\n    gm.account: {}\n    gm_account: {}\n",
            "master.players: 'gm.account' and 'gm_account' are one account "
            "(gm_account); keep one",
        ),
        ("  players: [a]\n", "master.players: expected key: value pairs"),
        ("  players:\n    a: 3\n", "master.players.a: expected key: value pairs"),
    ],
)
def test_errors(tmp_path, text, message):
    with pytest.raises(ScribeError, match=re.escape(f"scribe.yaml: {message}")):
        load(tmp_path, text)


def test_voice_step_used_only_by_a_player(tmp_path):
    cfg = load(
        tmp_path,
        "  voice:\n    hiss: {type: lowpass, f: 7000}\n"
        "  players:\n    a:\n      chain: [hiss, loudness]\n",
    )
    assert player(cfg, "a")[0].type == "lowpass"


def test_repeating_the_same_type_is_allowed(tmp_path):
    cfg = load(tmp_path, "  voice:\n    comp: {type: acompressor, ratio: 6}\n")
    assert cfg.voice[2].options["ratio"] == 6


def test_stereo_voices(tmp_path):
    music = Track("Kenku_FM_1234", "music", Path("3-Kenku_FM_1234.flac"))
    cfg = defaults()
    assert not cfg.keeps_stereo(Track("a", "player", Path("1-a.flac")))
    assert cfg.keeps_stereo(music)
    cfg = load(tmp_path, "  players:\n    pro.mic:\n      stereo: true\n")
    assert cfg.keeps_stereo(Track("pro.mic", "player", Path("1-pro.mic.flac")))
    assert not cfg.keeps_stereo(Track("other", "player", Path("2-other.flac")))
    same = [(s.name, s.type, s.options) for s in cfg.voice]
    assert [(s.name, s.type, s.options) for s in player(cfg, "pro.mic")] == same
    cfg = load(
        tmp_path,
        "  voice:\n    stereo: true\n  players:\n    a:\n      stereo: false\n",
    )
    assert cfg.keeps_stereo(Track("b", "player", Path("1-b.flac")))
    assert not cfg.keeps_stereo(Track("a", "player", Path("2-a.flac")))


def test_unknown_players(tmp_path):
    cfg = load(
        tmp_path,
        "  players:\n    gm.one:\n      comp: {ratio: 4}\n"
        "    absent:\n      comp: {ratio: 2}\n",
    )
    assert unknown_players(cfg, ["gm.one", "other"]) == ["absent"]
    assert unknown_players(cfg, ["gm_one", "absent"]) == []
