"""Tests for loading the built-in defaults and a config file."""

import textwrap
from pathlib import Path

import pytest

from scribe_rpg.config import load_config
from scribe_rpg.errors import ScribeError
from scribe_rpg.transcribe.glossary import load_glossary


def write(folder: Path, text: str, name: str = "scribe.yaml") -> Path:
    path = folder / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_defaults_only():
    cfg = load_config()
    assert set(cfg.settings) == {"input", "transcribe", "master"}
    assert cfg.settings["transcribe"]["whisper"]["model"] == "large-v3"
    assert cfg.settings["transcribe"]["whisper"]["language"] == "en"
    assert cfg.settings["transcribe"]["lines"]["pause"] == 1.5
    assert cfg.settings["input"]["music_patterns"] == ["kenku"]
    assert cfg.speakers == {}
    assert cfg.glossary_path is None


def test_defaults_cover_english_and_polish_phrases():
    phrases = load_config().settings["transcribe"]["cleanup"]["phrases"]
    assert "thanks for watching" in phrases
    assert "dzięki za oglądanie" in phrases


def test_partial_override_merges_recursively(tmp_path):
    cfg = load_config(write(tmp_path, "transcribe:\n  whisper:\n    batch_size: 2\n"))
    whisper = cfg.settings["transcribe"]["whisper"]
    assert whisper["batch_size"] == 2
    assert whisper["model"] == "large-v3"
    assert cfg.settings["transcribe"]["lines"]["pause"] == 1.5


def test_list_replaces_not_appends(tmp_path):
    cfg = load_config(write(tmp_path, "input:\n  music_patterns: [kenku, jukebox]\n"))
    assert cfg.settings["input"]["music_patterns"] == ["kenku", "jukebox"]


def test_unknown_key(tmp_path):
    with pytest.raises(
        ScribeError,
        match=r"scribe\.yaml: transcribe\.whisper\.bach_size: unknown key",
    ):
        load_config(write(tmp_path, "transcribe:\n  whisper:\n    bach_size: 2\n"))


def test_unknown_top_level_key(tmp_path):
    with pytest.raises(ScribeError, match=r"scribe\.yaml: speaker: unknown key"):
        load_config(write(tmp_path, "speaker:\n  gm.account: Game Master\n"))


def test_wrong_type(tmp_path):
    with pytest.raises(ScribeError, match=r"transcribe\.lines\.pause"):
        load_config(write(tmp_path, "transcribe:\n  lines:\n    pause: long\n"))


def test_section_must_be_mapping(tmp_path):
    with pytest.raises(
        ScribeError, match=r"transcribe\.whisper: expected key: value pairs"
    ):
        load_config(write(tmp_path, "transcribe:\n  whisper: large-v3\n"))


def test_transcribe_section_must_be_mapping(tmp_path):
    with pytest.raises(ScribeError, match=r"transcribe: expected key: value pairs"):
        load_config(write(tmp_path, "transcribe: yes please\n"))


def test_int_accepted_for_float(tmp_path):
    cfg = load_config(write(tmp_path, "transcribe:\n  lines:\n    pause: 2\n"))
    assert cfg.settings["transcribe"]["lines"]["pause"] == 2


def test_bool_not_accepted_for_number(tmp_path):
    with pytest.raises(ScribeError, match=r"transcribe\.lines\.pause"):
        load_config(write(tmp_path, "transcribe:\n  lines:\n    pause: true\n"))


def test_yaml_error_names_file_and_line(tmp_path):
    path = write(tmp_path, "transcribe:\n  whisper:\n    model: [unclosed\n")
    with pytest.raises(ScribeError, match=r"scribe\.yaml:\d+"):
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
    cfg = load_config(write(sub, "transcribe:\n  glossary: names.txt\n"))
    assert cfg.glossary_path == sub / "names.txt"


def test_glossary_at_top_level_is_unknown(tmp_path):
    with pytest.raises(ScribeError, match=r"scribe\.yaml: glossary: unknown key"):
        load_config(write(tmp_path, "glossary: names.txt\n"))


def test_glossary_option_wins(tmp_path):
    other = tmp_path / "other.txt"
    path = write(tmp_path, "transcribe:\n  glossary: names.txt\n")
    cfg = load_config(path, glossary=other)
    assert cfg.glossary_path == other


def test_keep_fillers_flag():
    cleanup = load_config(keep_fillers=True).settings["transcribe"]["cleanup"]
    assert cleanup["keep_fillers"] is True


def test_each_load_gets_fresh_defaults():
    load_config(keep_fillers=True)
    assert load_config().settings["transcribe"]["cleanup"]["keep_fillers"] is False


def test_empty_config_file(tmp_path):
    assert load_config(write(tmp_path, "")).settings == load_config().settings


def test_speakers_only_config_keeps_all_defaults(tmp_path):
    cfg = load_config(write(tmp_path, "speakers:\n  gm.account: Game Master\n"))
    assert cfg.settings == load_config().settings
    assert cfg.speakers == {"gm_account": "Game Master"}


def test_top_level_must_be_mapping(tmp_path):
    with pytest.raises(ScribeError, match="top level"):
        load_config(write(tmp_path, "- just\n- a list\n"))


def test_whisper_defaults_work_on_gpu_and_cpu():
    whisper = load_config().settings["transcribe"]["whisper"]
    assert whisper["compute_type"] == "default"  # WhisperX: float16 GPU, float32 CPU
    assert whisper["align_model"] is None  # WhisperX picks one by language


@pytest.mark.parametrize("value", ["org/wav2vec2-model", "null"])
def test_optional_setting_accepts_text_or_null(tmp_path, value):
    text = f"transcribe:\n  whisper:\n    align_model: {value}\n"
    whisper = load_config(write(tmp_path, text)).settings["transcribe"]["whisper"]
    assert whisper["align_model"] == (None if value == "null" else value)


def test_optional_setting_rejects_other_types(tmp_path):
    text = "transcribe:\n  whisper:\n    align_model: 5\n"
    with pytest.raises(ScribeError, match=r"whisper\.align_model: expected text"):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("whisper", "batch_size", 0),
        ("whisper", "batch_size", 2.5),
        ("whisper", "prompt_max_tokens", -1),
        ("cleanup", "frame_ms", 0),
        ("cleanup", "energy_db", 3),
        ("cleanup", "voiced_ratio", 2),
        ("cleanup", "voiced_ratio", ".nan"),
        ("cleanup", "phrase_max_words", -1),
        ("cleanup", "loop_max_repeat", 0),
        ("lines", "pause", -1),
        ("bleed", "similarity", 1.5),
        ("bleed", "min_overlap", -0.1),
        ("bleed", "min_words", 0),
    ],
)
def test_number_out_of_range(tmp_path, section, key, value):
    path = write(tmp_path, f"transcribe:\n  {section}:\n    {key}: {value}\n")
    with pytest.raises(
        ScribeError, match=rf"scribe\.yaml: transcribe\.{section}\.{key}: "
    ):
        load_config(path)


def test_numbers_at_range_ends_accepted(tmp_path):
    text = """\
        transcribe:
          whisper: {batch_size: 1, prompt_max_tokens: 0}
          cleanup: {energy_db: 0, voiced_ratio: 1, frame_ms: 12.5, loop_max_repeat: 1}
          lines: {pause: 0}
          bleed: {similarity: 0, min_overlap: 1}
        """
    load_config(write(tmp_path, text))


@pytest.mark.parametrize(
    "text",
    [
        "input:\n  music_patterns: [kenku, 2137]\n",
        "input:\n  music_patterns: [kenku, '']\n",
        "transcribe:\n  cleanup:\n    phrases: [thanks for watching, 404]\n",
    ],
)
def test_list_items_must_be_text(tmp_path, text):
    with pytest.raises(ScribeError, match=r"scribe\.yaml: .*: item 2: expected text"):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize("device", ["cuda:0", "gpu", "CPU"])
def test_device_is_cuda_or_cpu(tmp_path, device):
    path = write(tmp_path, f"transcribe:\n  whisper:\n    device: {device}\n")
    with pytest.raises(ScribeError, match=r"whisper\.device: expected cuda or cpu"):
        load_config(path)


def test_two_speaker_keys_for_one_account(tmp_path):
    path = write(tmp_path, "speakers:\n  a.b: One\n  a b: Two\n")
    with pytest.raises(
        ScribeError, match=r"speakers: 'a\.b' and 'a b' are one account"
    ):
        load_config(path)


@pytest.mark.parametrize("name", ["scribe.yaml", "glossary.txt"])
def test_file_not_utf8_names_the_line(tmp_path, name):
    (tmp_path / name).write_bytes("# ok\nmiasto: Łódź\n".encode("cp1250"))
    with pytest.raises(ScribeError, match=rf"{name}:2: not UTF-8; save it as UTF-8"):
        if name == "glossary.txt":
            load_glossary(tmp_path / name)
        else:
            load_config(tmp_path / name)
