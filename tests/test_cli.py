"""Tests for the command line: what every command shares, and `scribe init`."""

import subprocess
from pathlib import Path

import scribeio
from scribeio.transcribe import session_json

from helpers import (
    craig_zip,
    invoke,
    raw_session,
)


def test_version_option():
    result = invoke("--version")
    assert result.exit_code == 0
    assert scribeio.__version__ in result.output


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
    config = tmp_path / "cfg" / "scribe.yaml"
    assert f'  scribe master "{zip_path}" -c "{config}"\n' in result.output
    assert f'  scribe transcribe "{zip_path}" -c "{config}"\n' in result.output


def test_cli_import_does_not_load_numpy():
    # scribe merge must work without numpy loaded (fresh interpreter: the
    # test session itself has numpy imported already)
    import sys

    code = "import sys, scribeio.cli; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr


def test_master_modules_do_not_load_whisperx_or_torch():
    # scribe master must work without the [transcribe] extra
    import sys

    import scribeio.master as package

    names = sorted(p.stem for p in Path(package.__file__).parent.glob("*.py"))
    code = (
        "import importlib, sys\n"
        f"for name in {names!r}:\n"
        "    importlib.import_module(f'scribeio.master.{name}')\n"
        "print(sorted({'whisperx', 'torch'} & set(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stderr


def test_init_only_master(tmp_path):
    zip_path = craig_zip(tmp_path / "rec")
    result = invoke("init", zip_path, "--only", "master")
    assert result.exit_code == 0, result.output
    assert "created scribe.yaml in" in result.output
    assert not (tmp_path / "rec" / "glossary.txt").exists()
    next_steps = result.output.split("Next:")[1]
    assert "scribe master" in next_steps and "scribe transcribe" not in next_steps


def test_init_both_names_both_commands(tmp_path):
    result = invoke("init", "-o", tmp_path)
    next_steps = result.output.split("Next:")[1]
    assert "scribe master" in next_steps and "scribe transcribe" in next_steps


def test_init_full(tmp_path):
    result = invoke("init", "-o", tmp_path, "--full", "--only", "transcribe")
    assert result.exit_code == 0, result.output
    text = (tmp_path / "scribe.yaml").read_text(encoding="utf-8")
    assert "phrases:" in text and "master:" not in text


def test_init_language_needs_transcription(tmp_path):
    result = invoke("init", "-o", tmp_path, "--only", "master", "--language", "pl")
    assert result.exit_code == 2 and "--language" in result.output


def test_init_from_json_hints_merge(tmp_path):
    path = tmp_path / "2026-01-01_transcript.json"
    session_json.write(path, raw_session([]))
    result = invoke("init", path)
    assert result.exit_code == 0, result.output
    next_steps = result.output.split("Next:")[1]
    assert f'scribe merge "{path}"' in next_steps
    assert "scribe transcribe" not in next_steps
    assert "scribe master <recording>" in next_steps


def test_check_is_listed_with_its_short_help():
    output = invoke("--help").output
    assert "Show what scribe needs and what it finds (offline)." in output


def test_check_prints_report_and_exits_with_its_code(monkeypatch):
    from scribeio import check_report

    monkeypatch.setattr(
        check_report, "report", lambda config, cwd, online: (["a", "b"], 1)
    )
    result = invoke("check")
    assert result.exit_code == 1
    assert result.output == "a\nb\n"


def test_check_builds_the_report_behind_a_busy_indicator(monkeypatch):
    from contextlib import contextmanager

    from scribeio import check_report
    from scribeio.progress import Progress

    busy = []

    @contextmanager
    def fake_busy(self, label):
        busy.append(label)
        yield
        busy.append("done")

    def report(config, cwd, online):
        assert busy and busy[-1] != "done"  # built while the indicator runs
        return ["a"], 0

    monkeypatch.setattr(Progress, "busy", fake_busy)
    monkeypatch.setattr(check_report, "report", report)
    assert invoke("check").output == "a\n"
    assert busy[-1] == "done"


def test_utf8_output_for_a_file_with_a_legacy_encoding():
    import io

    from scribeio import cli

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1250")
    cli._utf8_output(stream)
    stream.write("✔ ✘ ⚠ ℹ")
    stream.flush()
    assert raw.getvalue().decode("utf-8") == "✔ ✘ ⚠ ℹ"


def test_check_online_option(monkeypatch):
    from scribeio import check_report

    seen = []
    monkeypatch.setattr(
        check_report,
        "report",
        lambda config, cwd, online: (seen.append(online), ([], 0))[1],
    )
    assert invoke("check", "--online").exit_code == 0
    assert invoke("check").exit_code == 0
    assert seen == [True, False]
    assert "test the model sources" in invoke("check", "--help").output


def fake_colorama(monkeypatch, platform):
    """Pretend to run on `platform` with a colorama that records its calls."""
    import sys
    import types

    calls = []
    module = types.ModuleType("colorama")
    module.just_fix_windows_console = lambda: calls.append("fixed")
    monkeypatch.setitem(sys.modules, "colorama", module)
    monkeypatch.setattr(sys, "platform", platform)
    return calls


def test_windows_console_shows_colours(monkeypatch):
    # the old Windows console prints colour codes as text unless switched on
    calls = fake_colorama(monkeypatch, "win32")
    assert invoke("init", "--help").exit_code == 0
    assert calls == ["fixed"]


def test_colorama_not_used_elsewhere(monkeypatch):
    calls = fake_colorama(monkeypatch, "linux")
    assert invoke("init", "--help").exit_code == 0
    assert calls == []
