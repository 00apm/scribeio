"""Tests for the `scribe check` report, with the checks and caches faked."""

import re
from pathlib import Path

import pytest

import scribeio
from scribeio import check_report, checks
from scribeio.checks import Item
from scribeio.errors import ScribeError
from scribeio.transcribe import models

NONE_YET = "none yet: downloaded on your first transcription"


@pytest.fixture
def ready(monkeypatch, tmp_path):
    """Everything installed and both models of the default settings on disk."""
    state = {"encoder_calls": []}

    def encoder_item(fmt, bitrate):
        state["encoder_calls"].append((fmt, bitrate))
        return Item(f"{fmt} encoder", "ok")

    def needed(cfg):
        if cfg["model"] == "no-such-model":
            raise ScribeError("unknown Whisper model 'no-such-model': invalid")
        return [
            models.Model("Whisper model", cfg["model"], "huggingface", True),
            models.Model(
                "alignment model", "org/x", "huggingface", False,
                language=cfg["language"],
            ),
        ]  # fmt: skip

    monkeypatch.setattr(
        checks, "ffmpeg_item", lambda: Item("ffmpeg", "ok", "7.1.1  C:\\ff.exe")
    )
    monkeypatch.setattr(checks, "encoder_item", encoder_item)
    monkeypatch.setattr(
        checks, "whisperx_item", lambda: Item("WhisperX", "ok", "3.8.6")
    )
    monkeypatch.setattr(
        checks, "torch_item", lambda device: Item("PyTorch", "ok", "2.8.0, CUDA build")
    )
    monkeypatch.setattr(
        checks, "gpu_item", lambda device: Item("GPU", "ok", message="NVIDIA RTX, 8 GB")
    )
    monkeypatch.setattr(models, "cache_folder", lambda: tmp_path / "hf")
    monkeypatch.setattr(models, "cached_whisper_models", lambda: ["large-v3", "medium"])
    monkeypatch.setattr(models, "local_alignment", lambda: (["en", "pl"], []))
    monkeypatch.setattr(models, "needed", needed)
    return state


def run(tmp_path, config=None):
    lines, code = check_report.report(config, tmp_path, online=False)
    return "\n".join(lines), code


def write_config(folder: Path, text: str) -> Path:
    path = folder / "scribe.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_all_ready_exit_0(ready, tmp_path):
    text, code = run(tmp_path)
    assert code == 0
    assert re.search(r"^Audio mastering +ready$", text, re.M)
    assert re.search(r"^Transcription +ready$", text, re.M)
    assert "  ✔ ffmpeg 7.1.1  C:\\ff.exe" in text
    assert "  ✔ GPU: NVIDIA RTX, 8 GB" in text


def test_verdict_column_matches_the_spec(ready, tmp_path):
    text, _ = run(tmp_path)
    assert "Audio mastering" + " " * 38 + "ready" in text.splitlines()


BROKEN = {
    "ffmpeg": ("ffmpeg_item", lambda *a: Item("ffmpeg", "missing", "", "m", "ffmpeg")),
    "encoder": (
        "encoder_item",
        lambda *a: Item("mp3 encoder", "missing", "", "m", "encoder"),
    ),
    "torch": (
        "torch_item",
        lambda *a: Item("PyTorch", "missing", "2.8.0", "m", "pytorch-cuda"),
    ),
    "gpu": ("gpu_item", lambda *a: Item("GPU", "missing", "", "m", "gpu")),
    "whisperx_broken": (
        "whisperx_item",
        lambda *a: Item("WhisperX", "missing", "", "installed but", "transcription"),
    ),
}


@pytest.mark.parametrize("broken", list(BROKEN))
def test_not_ready_exit_1(ready, broken, monkeypatch, tmp_path):
    name, fake = BROKEN[broken]
    monkeypatch.setattr(checks, name, fake)
    lines, code = check_report.report(None, tmp_path, online=False)
    assert code == 1
    [i] = [i for i, s in enumerate(lines) if s.startswith("  ✘ ")]
    assert lines[i + 1].startswith("    see https://")


def test_ffmpeg_missing_skips_encoder_and_blocks_transcription(
    ready, monkeypatch, tmp_path
):
    monkeypatch.setattr(checks, "ffmpeg_item", BROKEN["ffmpeg"][1])
    text, code = run(tmp_path)
    assert code == 1 and ready["encoder_calls"] == []
    assert "encoder" not in text
    assert re.search(r"^Audio mastering +not ready$", text, re.M)
    assert re.search(r"^Transcription +not ready$", text, re.M)


def test_whisperx_not_installed_is_exit_0(ready, monkeypatch, tmp_path):
    monkeypatch.setattr(
        checks,
        "whisperx_item",
        lambda: Item(
            "WhisperX", "missing", "", checks.WHISPERX_NOT_INSTALLED, "transcription"
        ),
    )
    lines, code = check_report.report(None, tmp_path, online=False)
    text = "\n".join(lines)
    assert code == 0
    assert re.search(
        r"^Transcription +not installed \(audio mastering only\)$", text, re.M
    )
    assert "(audio mastering works)" in checks.WHISPERX_NOT_INSTALLED
    i = lines.index("  ℹ WhisperX: " + checks.WHISPERX_NOT_INSTALLED)
    assert lines[i + 1] == "    see " + checks.guide_url("transcription")
    assert "PyTorch" not in text and "Models (" not in text


def test_cpu_device_in_config_makes_gpu_info(ready, monkeypatch, tmp_path):
    def torch_item(device):
        if device == "cpu":
            return Item("PyTorch", "info", "2.8.0", "set to run on the CPU: very slow")
        return Item("PyTorch", "missing", "2.8.0", "m", "pytorch-cuda")

    monkeypatch.setattr(checks, "torch_item", torch_item)
    monkeypatch.setattr(
        checks,
        "gpu_item",
        lambda device: Item("GPU", "info", "", "set to run on the CPU: very slow"),
    )
    config = write_config(tmp_path, "transcribe:\n  whisper:\n    device: cpu\n")
    text, code = run(tmp_path, config)
    assert code == 0
    assert "  ℹ PyTorch 2.8.0: set to run on the CPU" in text
    assert "  ℹ GPU: set to run on the CPU" in text


def test_cpu_only_torch_on_cuda_skips_the_gpu_line(ready, monkeypatch, tmp_path):
    monkeypatch.setattr(checks, "torch_item", BROKEN["torch"][1])
    text, code = run(tmp_path)
    assert code == 1 and "GPU" not in text


def test_ffmpeg_8_info_line(ready, monkeypatch, tmp_path):
    monkeypatch.setattr(
        checks, "ffmpeg_item", lambda: Item("ffmpeg", "ok", "9.0.2  C:\\ff.exe")
    )
    text, code = run(tmp_path)
    assert code == 0
    assert "  ℹ ffmpeg 9: WhisperX's audio library prints a warning; harmless" in text


def test_models_listed(ready, tmp_path):
    lines, _ = check_report.report(None, tmp_path, online=False)
    assert f"Models ({tmp_path / 'hf'})" in lines
    assert "  Whisper:   large-v3, medium" in lines
    assert "  Alignment: en, pl" in lines


def test_no_models_yet(ready, monkeypatch, tmp_path):
    monkeypatch.setattr(models, "cached_whisper_models", lambda: [])
    monkeypatch.setattr(models, "local_alignment", lambda: ([], []))
    lines, _ = check_report.report(None, tmp_path, online=False)
    assert "  Whisper:   " + NONE_YET in lines
    assert "  Alignment: " + NONE_YET in lines


def test_damaged_alignment(ready, monkeypatch, tmp_path):
    monkeypatch.setattr(models, "local_alignment", lambda: (["pl"], ["en"]))
    lines, code = check_report.report(None, tmp_path, online=False)
    assert code == 0
    assert "  ⚠ alignment en: damaged; replaced on the next transcription" in lines


def test_settings_from_cwd(ready, tmp_path):
    path = write_config(tmp_path, "transcribe:\n  whisper:\n    language: pl\n")
    lines, _ = check_report.report(None, tmp_path, online=False)
    assert f"  Settings checked: {path.resolve()}" in lines
    assert (
        "    needs large-v3 ✔, alignment pl ↓ (downloaded on first transcription)"
        in lines
    )


def test_settings_from_dash_c(ready, tmp_path):
    write_config(tmp_path, "")
    other = tmp_path / "other"
    other.mkdir()
    path = write_config(other, "")
    lines, _ = check_report.report(path, tmp_path, online=False)
    assert f"  Settings checked: {path.resolve()}" in lines
    assert f"  Settings checked: {(tmp_path / 'scribe.yaml').resolve()}" not in lines


def test_no_settings(ready, tmp_path):
    lines, code = check_report.report(None, tmp_path, online=False)
    assert code == 0
    assert f"  No scribe.yaml in {tmp_path} (the folder this terminal is in)." in lines
    # with the recording: a bare `scribe init` writes example accounts
    assert (
        "  To set up a session: go to the folder with your recording and run "
        'scribe init "<your recording>.zip"' in lines
    )
    assert not any(re.search(r"\bhere\b", s) for s in lines)


def test_no_settings_without_whisperx(ready, monkeypatch, tmp_path):
    monkeypatch.setattr(checks, "whisperx_item", BROKEN["whisperx_broken"][1])
    lines, _ = check_report.report(None, tmp_path, online=False)
    assert f"  No scribe.yaml in {tmp_path} (the folder this terminal is in)." in lines


def test_settings_do_not_load(ready, tmp_path):
    path = write_config(tmp_path, "master: [\n")
    lines, code = check_report.report(None, tmp_path, online=False)
    text = "\n".join(lines)
    assert code == 0
    [bad] = [s for s in lines if s.startswith("  ⚠ Settings checked: ")]
    assert str(path) in bad and "YAML" in bad
    assert "Models (" in text and re.search(r"^Transcription +ready$", text, re.M)


def test_unknown_model_in_settings(ready, tmp_path):  # Review Focus 1
    write_config(tmp_path, "transcribe:\n  whisper:\n    model: no-such-model\n")
    lines, code = check_report.report(None, tmp_path, online=False)
    assert code == 0
    [bad] = [s for s in lines if s.startswith("  ⚠ Settings checked: ")]
    assert "unknown Whisper model" in bad
    assert not any(s.startswith("    needs") for s in lines)


def test_encoder_uses_the_configs_format(ready, tmp_path):
    write_config(tmp_path, "master:\n  format: opus\n")
    run(tmp_path)
    assert ready["encoder_calls"] == [("opus", "48k")]


def test_encoder_uses_the_configs_bitrate(ready, tmp_path):
    write_config(tmp_path, "master:\n  bitrate: 128k\n")
    run(tmp_path)
    assert ready["encoder_calls"] == [("mp3", "128k")]


def test_header_route(monkeypatch, tmp_path):
    monkeypatch.setattr(check_report.sys, "prefix", str(tmp_path))
    assert check_report.header().endswith(f"(in {tmp_path})")
    assert check_report.header().startswith(f"scribeio {scribeio.__version__}, ")
    (tmp_path / "uv-receipt.toml").write_text("")
    assert check_report.header().endswith("(installed with uv)")


# --- --online -----------------------------------------------------------------


@pytest.fixture
def network(monkeypatch):
    """Model sizes and reachability faked; records what was asked."""
    state = {"sizes": [], "reached": [], "size": 1288490188, "down": set()}

    def size(model):
        state["sizes"].append(model.name)
        return state["size"]

    def reach(host, url, timeout=10.0):
        state["reached"].append(host)
        if host in state["down"]:
            return Item(host, "warning", "", "not reachable: timed out", "connection")
        return Item(host, "ok")

    monkeypatch.setattr(models, "size", size)
    monkeypatch.setattr(checks, "reach", reach)
    return state


def online(tmp_path, config=None):
    lines, code = check_report.report(config, tmp_path, online=True)
    return lines, code


def test_offline_never_touches_network(ready, monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError("network used offline")

    monkeypatch.setattr(checks, "urlopen", no_network)
    monkeypatch.setattr(checks, "reach", no_network)
    monkeypatch.setattr(models, "size", no_network)
    write_config(tmp_path, "")
    lines, code = check_report.report(None, tmp_path, online=False)
    assert code == 0 and "Connection" not in lines


def test_online_sizes_of_missing_models(ready, network, tmp_path):
    write_config(tmp_path, "transcribe:\n  whisper:\n    language: pl\n")
    lines, code = online(tmp_path)
    assert code == 0
    assert (
        "    needs large-v3 ✔, alignment pl ↓ (1.2 GB, downloaded on first "
        "transcription)" in lines
    )
    assert network["sizes"] == ["org/x"]  # models on disk are not looked up


def test_online_size_lookup_failing(ready, network, tmp_path):
    network["size"] = None
    write_config(tmp_path, "")
    lines, _ = online(tmp_path)
    [needs] = [s for s in lines if s.startswith("    needs")]
    assert "(size unknown, downloaded on first transcription)" in needs


def test_online_connection_block(ready, network, tmp_path):
    write_config(tmp_path, "")
    lines, _ = online(tmp_path)
    i = lines.index("Connection")
    assert lines[i - 1] == "" and lines[i + 1] == "  ✔ huggingface.co"
    assert network["reached"] == ["huggingface.co"]
    assert not any(checks.CONNECTION_INFO in s for s in lines)


def test_online_pytorch_host_only_for_torchaudio_model(
    ready, network, monkeypatch, tmp_path
):
    def needed(cfg):
        return [
            models.Model("Whisper model", "large-v3", "huggingface", True),
            models.Model("alignment model", "WAV2VEC2", "torchaudio", True),
        ]

    monkeypatch.setattr(models, "needed", needed)
    write_config(tmp_path, "")
    lines, _ = online(tmp_path)
    assert network["reached"] == ["huggingface.co", "download.pytorch.org"]
    assert "  ✔ download.pytorch.org" in lines


def test_online_unreachable_is_exit_0_with_info(ready, network, tmp_path):
    network["down"] = {"huggingface.co"}
    lines, code = online(tmp_path)
    assert code == 0
    i = lines.index("  ⚠ huggingface.co: not reachable: timed out")
    assert lines[i + 1] == "  ℹ " + checks.CONNECTION_INFO
    assert lines[i + 2] == "    see " + checks.guide_url("connection")


def test_online_without_config_checks_reachability_only(ready, network, tmp_path):
    lines, _ = online(tmp_path)
    assert network["sizes"] == [] and network["reached"] == ["huggingface.co"]


def test_online_without_whisperx_checks_reachability_only(
    ready, network, monkeypatch, tmp_path
):
    monkeypatch.setattr(checks, "whisperx_item", BROKEN["whisperx_broken"][1])
    write_config(tmp_path, "")
    lines, _ = online(tmp_path)
    assert network["sizes"] == [] and network["reached"] == ["huggingface.co"]


def test_settings_heading_without_whisperx(ready, monkeypatch, tmp_path):
    monkeypatch.setattr(checks, "whisperx_item", BROKEN["whisperx_broken"][1])
    path = write_config(tmp_path, "")
    lines, _ = check_report.report(None, tmp_path, online=False)
    i = lines.index("Settings")
    assert (
        lines[i - 1] == "" and lines[i + 1] == f"  Settings checked: {path.resolve()}"
    )


def test_no_settings_heading_with_the_models_block(ready, tmp_path):
    lines, _ = check_report.report(None, tmp_path, online=False)
    assert "Settings" not in lines


def test_settings_error_names_the_file_once(ready, tmp_path, monkeypatch):
    path = write_config(tmp_path, "master: [\n")
    monkeypatch.chdir(tmp_path)
    lines, _ = check_report.report(Path("scribe.yaml"), tmp_path, online=False)
    [bad] = [s for s in lines if s.startswith("  ⚠ Settings checked: ")]
    assert bad.startswith(f"  ⚠ Settings checked: {path}:2: invalid YAML")
    assert bad.count("scribe.yaml") == 1


def test_unknown_model_error_shows_the_full_path(ready, tmp_path, monkeypatch):
    path = write_config(tmp_path, "transcribe:\n  whisper:\n    model: no-such-model\n")
    monkeypatch.chdir(tmp_path)
    lines, _ = check_report.report(Path("scribe.yaml"), tmp_path, online=False)
    assert f"  ⚠ Settings checked: {path}: unknown Whisper model" in "\n".join(lines)


def test_settings_path_kept_as_given_not_resolved(ready, tmp_path, monkeypatch):
    # resolve() turns a mapped network drive (Z:\) into \server\share\...
    path = write_config(tmp_path, "")
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: Path("X:/unc"))
    lines, _ = check_report.report(None, tmp_path, online=False)
    assert f"  Settings checked: {path.absolute()}" in lines


def test_online_no_size_lookups_when_the_source_is_unreachable(
    ready, network, tmp_path
):
    network["down"] = {"huggingface.co"}
    write_config(tmp_path, "")
    lines, code = online(tmp_path)
    assert code == 0 and network["sizes"] == []
    [needs] = [s for s in lines if s.startswith("    needs")]
    assert "(size unknown, downloaded on first transcription)" in needs
    assert lines.index("Connection") > lines.index(needs)  # layout unchanged


def test_report_shows_each_section_as_it_is_done(ready, monkeypatch, tmp_path):
    # a first run spends minutes loading WhisperX and PyTorch: the instant
    # parts are shown first, the slow ones behind a named indicator
    from contextlib import contextmanager

    events = []
    active = []

    @contextmanager
    def busy(label):
        events.append(("busy", label))
        active.append(label)
        yield
        active.pop()

    def whisperx_item():
        events.append(("whisperx", list(active)))
        return Item("WhisperX", "ok", "3.8.6")

    monkeypatch.setattr(checks, "whisperx_item", whisperx_item)
    shown = []

    def show(lines):
        events.append(("show", lines[0]))
        shown.extend(lines)

    lines, _ = check_report.report(None, tmp_path, online=False, show=show, busy=busy)
    assert shown == lines
    assert events[0] == ("show", lines[0])  # header and audio before the wait
    assert events[1] == ("busy", check_report.LOADING)
    assert events[2] == ("whisperx", [check_report.LOADING])
    assert any(e[0] == "show" and e[1].startswith("Transcription") for e in events)
