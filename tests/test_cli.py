"""Tests for the command line: what every command shares, and `scribe init`."""

import subprocess

import scribe_rpg

from helpers import (
    craig_zip,
    invoke,
)


def test_version_option():
    result = invoke("--version")
    assert result.exit_code == 0
    assert scribe_rpg.__version__ in result.output


def test_help_texts():
    assert "Craig recording" in invoke("transcribe", "--help").output
    assert "without the GPU" in invoke("merge", "--help").output


def test_usage_error_exit_2():
    assert invoke("transcribe").exit_code == 2


def test_init_without_recording(tmp_path):
    result = invoke("init", "-o", tmp_path / "cfg")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "cfg" / "scribe.yaml").exists()
    assert (tmp_path / "cfg" / "glossary.txt").exists()
    assert "created scribe.yaml, glossary.txt" in result.output


def test_init_from_recording_prefills_speakers(tmp_path):
    zip_path = craig_zip(tmp_path / "rec")
    result = invoke("init", zip_path, "--language", "pl")
    assert result.exit_code == 0, result.output
    text = (tmp_path / "rec" / "scribe.yaml").read_text(encoding="utf-8")
    assert '"gm.account": "gm.account"' in text
    assert "music bots (not transcribed): Kenku FM#1234" in text
    assert "language: pl" in text
    assert "scribe transcribe" in result.output  # next step hint


def test_init_refuses_to_overwrite(tmp_path):
    assert invoke("init", "-o", tmp_path).exit_code == 0
    result = invoke("init", "-o", tmp_path)
    assert result.exit_code == 1 and "--force" in result.output


def test_init_elsewhere_hints_config_option(tmp_path):
    zip_path = craig_zip(tmp_path / "rec")
    result = invoke("init", zip_path, "-o", tmp_path / "cfg")
    assert result.exit_code == 0, result.output
    assert "-c" in result.output and "scribe.yaml" in result.output.split("Next:")[1]


def test_cli_import_does_not_load_numpy():
    # scribe merge must work without numpy loaded (fresh interpreter: the
    # test session itself has numpy imported already)
    import sys

    code = "import sys, scribe_rpg.cli; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr
