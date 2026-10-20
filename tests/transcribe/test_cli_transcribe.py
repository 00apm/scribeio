"""Tests for `scribe transcribe` and `scribe merge`, with a fake transcriber."""

import json
import sys
import textwrap
import warnings
from datetime import datetime, timezone

import numpy as np
import pytest
from click.testing import CliRunner

from scribeio import checks, ffmpeg
from scribeio.cli import main
from scribeio.errors import ScribeError
from scribeio.transcribe import models, whisper

from helpers import (
    craig_dir,
    craig_unzipped,
    craig_zip,
    invoke,
    seg,
    silence,
    tone,
)

LOCAL_DATE = datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc).astimezone()
LOCAL_DATE = LOCAL_DATE.strftime("%Y-%m-%d")
VERSIONS = {"whisperx": "3.8.6", "torch": "2.8.0+cu128", "ffmpeg": "7.1"}
REAL_PREFLIGHT = whisper.preflight


class FakeTranscriber:
    """Same interface as whisper.Transcriber; returns canned segments."""

    calls: list[str] = []
    fail: Exception | None = None
    extra_warnings: list[str] = []
    library_warnings: list[str] = []  # emitted like pyannote's, on loading

    def __init__(self, cfg, names):
        self.cfg = cfg
        self.initial_prompt = ", ".join(names)
        self.prompt_cut: list[str] = []
        self.warnings = list(FakeTranscriber.extra_warnings)
        for text in FakeTranscriber.library_warnings:
            warnings.warn(text, stacklevel=1)

    def transcribe(self, audio, account, on_progress=None, on_phase=None):
        if on_phase:
            on_phase("transcribing")
        FakeTranscriber.calls.append(account)
        if FakeTranscriber.fail:
            raise FakeTranscriber.fail
        if on_progress:
            on_progress(50.0)
            on_progress(100.0)
        if account == "gm_account":
            return [
                seg("gm_account", 1.0, "Witajcie na sesji w Worgradzie"),
                seg("gm_account", 15.0, "Dzięki za oglądanie!"),  # over silence
            ]
        return [seg(account, 5.0, "Tak.")]

    def settings(self):
        return {
            **self.cfg,
            "initial_prompt": self.initial_prompt,
            "prompt_cut": self.prompt_cut,
        }


@pytest.fixture
def fake_whisper(monkeypatch):
    FakeTranscriber.calls = []
    FakeTranscriber.fail = None
    FakeTranscriber.extra_warnings = []
    FakeTranscriber.library_warnings = []
    audio = np.concatenate([tone(10.0), silence(10.0)])  # speech 0-10 s, then silence
    monkeypatch.setattr(whisper, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(whisper, "load_audio", lambda path: audio)
    monkeypatch.setattr(whisper, "preflight", lambda cfg: [])
    monkeypatch.setattr(whisper, "versions", lambda: dict(VERSIONS))
    FakeTranscriber.music_seconds = 15.0  # shorter than the voices
    monkeypatch.setattr(
        "scribeio.ffmpeg.duration", lambda path: FakeTranscriber.music_seconds
    )
    FakeTranscriber.models = [
        models.Model("Whisper model", "large-v3", "huggingface", True),
        models.Model("alignment model", "WAV2VEC2_ASR_BASE_960H", "torchaudio", True),
    ]
    FakeTranscriber.downloads = []
    FakeTranscriber.others = []
    monkeypatch.setattr(models, "needed", lambda cfg: FakeTranscriber.models)
    monkeypatch.setattr(models, "size", lambda m: 3_100_000_000)
    monkeypatch.setattr(models, "cached_whisper_models", lambda: FakeTranscriber.others)
    monkeypatch.setattr(
        models, "download", lambda m, progress: FakeTranscriber.downloads.append(m.name)
    )
    return FakeTranscriber


@pytest.fixture
def campaign(tmp_path):
    folder = tmp_path / "campaign"
    folder.mkdir()
    glossary = folder / "glossary.txt"
    glossary.write_text("Vorgrad\nWorgrad* -> Vorgrad\n", encoding="utf-8")
    config = folder / "scribe.yaml"
    config.write_text(
        textwrap.dedent(
            """\
            speakers:
              gm.account: Game Master
              player.one: Player One
            transcribe:
              glossary: glossary.txt
            """
        ),
        encoding="utf-8",
    )
    return config


def test_transcribe_writes_three_outputs(tmp_path, fake_whisper, campaign):
    zip_path = craig_zip(tmp_path / "rec")
    result = invoke("transcribe", zip_path, "-c", campaign)
    assert result.exit_code == 0, result.output
    out = tmp_path / "rec"
    md = (out / f"{LOCAL_DATE}_transcript.md").read_text(encoding="utf-8")
    assert "[00:00:01] Game Master: Witajcie na sesji w Vorgradzie" in md
    assert "[00:00:05] Player One: Tak." in md
    assert "Dzięki" not in md
    report = (out / f"{LOCAL_DATE}_transcript-report.md").read_text(encoding="utf-8")
    assert "Dzięki za oglądanie! (0% voiced)" in report
    assert "whisperx 3.8.6" in report
    data = json.loads(
        (out / f"{LOCAL_DATE}_transcript.json").read_text(encoding="utf-8")
    )
    assert data["whisper"]["initial_prompt"] == "Vorgrad"
    assert data["session"]["duration"] == 20.0
    assert data["versions"] == VERSIONS
    assert [t["transcribed"] for t in data["tracks"]] == [True, True, False]
    assert fake_whisper.calls == ["gm_account", "player_one"]  # music never


def test_duration_includes_the_music_track(tmp_path, fake_whisper, campaign):
    fake_whisper.music_seconds = 60.0  # the bot joined first, left last
    zip_path = craig_zip(tmp_path / "rec")
    result = invoke("transcribe", zip_path, "-c", campaign, "--name", "s")
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "rec" / "s_transcript.json").read_text("utf-8"))
    assert data["session"]["duration"] == 60.0


def test_unreadable_music_track_only_warns(
    tmp_path, fake_whisper, campaign, monkeypatch
):
    def fail(path):
        raise ScribeError("ffmpeg failed (duration x.flac): Invalid data")

    monkeypatch.setattr("scribeio.ffmpeg.duration", fail)
    zip_path = craig_zip(tmp_path / "rec")
    result = invoke("transcribe", zip_path, "-c", campaign, "--name", "s")
    assert result.exit_code == 0, result.output
    assert "Kenku_FM_1234: length unknown, not in the duration" in result.output
    data = json.loads((tmp_path / "rec" / "s_transcript.json").read_text("utf-8"))
    assert data["session"]["duration"] == 20.0


def test_outputs_contain_no_paths(tmp_path, fake_whisper, campaign):
    zip_path = craig_zip(tmp_path / "rec")
    out = tmp_path / "out"
    result = invoke("transcribe", zip_path, "-c", campaign, "-o", out, "--name", "s")
    assert result.exit_code == 0, result.output
    data = json.loads((out / "s_transcript.json").read_text(encoding="utf-8"))
    assert data["session"]["source"] == "session.zip"
    assert all("/" not in t["file"] and "\\" not in t["file"] for t in data["tracks"])
    for f in out.iterdir():
        text = f.read_text(encoding="utf-8")
        assert tmp_path.name not in text, f.name
        assert "scribe-" not in text, f.name  # temp folder prefix


def test_existing_outputs_need_force(tmp_path, fake_whisper, campaign):
    zip_path = craig_zip(tmp_path / "rec")
    assert invoke("transcribe", zip_path, "-c", campaign).exit_code == 0
    again = invoke("transcribe", zip_path, "-c", campaign)
    assert again.exit_code == 1
    assert "--force" in again.output and "Traceback" not in again.output
    assert invoke("transcribe", zip_path, "-c", campaign, "--force").exit_code == 0


def test_preflight_failure_stops_before_unzipping(
    tmp_path, fake_whisper, campaign, monkeypatch
):
    def fail(cfg):
        raise ScribeError("ffmpeg not found in PATH; install it")

    monkeypatch.setattr(whisper, "preflight", fail)
    result = invoke("transcribe", craig_zip(tmp_path / "rec"), "-c", campaign)
    assert result.exit_code == 1 and "ffmpeg not found" in result.output
    assert "open session.zip" not in result.output
    assert fake_whisper.calls == []


def test_whisperx_missing_points_to_scribe_check(
    tmp_path, fake_whisper, campaign, monkeypatch
):
    monkeypatch.setattr(whisper, "preflight", REAL_PREFLIGHT)
    monkeypatch.setitem(sys.modules, "whisperx", None)
    monkeypatch.setattr(checks.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(ffmpeg, "version", lambda: "7.1")
    result = invoke("transcribe", craig_zip(tmp_path / "rec"), "-c", campaign)
    assert result.exit_code == 1
    assert "WhisperX: not installed" in result.output
    assert result.output.rstrip().endswith(checks.CHECK_HINT)


def test_preflight_and_transcriber_warnings_shown(
    tmp_path, fake_whisper, campaign, monkeypatch
):
    monkeypatch.setattr(whisper, "preflight", lambda cfg: ["runs on cpu: very slow"])
    fake_whisper.extra_warnings = ["prompt length estimated"]
    result = invoke("transcribe", craig_zip(tmp_path / "rec"), "-c", campaign)
    assert result.exit_code == 0, result.output
    assert "very slow" in result.output and "prompt length estimated" in result.output


def test_run_warnings_reach_the_report_and_survive_merge(
    tmp_path, fake_whisper, campaign, monkeypatch
):
    monkeypatch.setattr(whisper, "preflight", lambda cfg: ["runs on cpu: very slow"])
    fake_whisper.extra_warnings = ["prompt length estimated"]
    zip_path = craig_zip(tmp_path / "rec")
    assert invoke("transcribe", zip_path, "-c", campaign, "--name", "s").exit_code == 0
    report = tmp_path / "rec" / "s_transcript-report.md"
    for _ in range(2):  # after transcribe, then after merge
        text = report.read_text(encoding="utf-8").split("## Warnings")[1]
        assert "- runs on cpu: very slow\n- prompt length estimated\n" in text
        result = invoke("merge", tmp_path / "rec" / "s_transcript.json", "-c", campaign)
        assert result.exit_code == 0, result.output


TORCHCODEC = (
    "\ntorchcodec is not installed correctly so built-in audio decoding will "
    "fail. Solutions are:\n* fix torchcodec installation."
)


def test_known_library_warnings_go_to_the_report(tmp_path, fake_whisper, campaign):
    fake_whisper.library_warnings = [TORCHCODEC]
    zip_path = craig_zip(tmp_path / "rec")
    result = invoke("transcribe", zip_path, "-c", campaign, "--name", "s")
    assert result.exit_code == 0, result.output
    assert "torchcodec" not in result.output
    assert "1 library warning hidden, listed in the report" in result.output
    report = (tmp_path / "rec" / "s_transcript-report.md").read_text("utf-8")
    assert (
        "- library warning: torchcodec is not installed correctly so built-in "
        "audio decoding will fail. Solutions are: * fix torchcodec installation.\n"
    ) in report


def test_no_hidden_line_without_library_warnings(tmp_path, fake_whisper, campaign):
    result = invoke("transcribe", craig_zip(tmp_path / "rec"), "-c", campaign)
    assert result.exit_code == 0, result.output
    assert "library warning" not in result.output


def test_hidden_library_warnings_shown_when_the_run_fails(
    tmp_path, fake_whisper, campaign
):
    fake_whisper.library_warnings = [TORCHCODEC]
    fake_whisper.fail = ScribeError("CUDA out of memory; try a smaller batch_size")
    result = invoke("transcribe", craig_zip(tmp_path / "rec"), "-c", campaign)
    assert result.exit_code == 1
    hidden = result.output.index("library warnings (may explain the error):")
    assert result.output.index("torchcodec is not installed") > hidden
    assert result.output.index("smaller batch_size") > hidden


def test_failure_mid_run_writes_nothing(tmp_path, fake_whisper, campaign):
    fake_whisper.fail = ScribeError("CUDA out of memory; try a smaller batch_size")
    out = tmp_path / "out"
    result = invoke(
        "transcribe", craig_zip(tmp_path / "rec"), "-c", campaign, "-o", out
    )
    assert result.exit_code == 1
    assert "smaller batch_size" in result.output and "Traceback" not in result.output
    assert not out.exists() or not any(out.iterdir())


def test_unknown_account_warns(tmp_path, fake_whisper):
    result = invoke("transcribe", craig_zip(tmp_path / "rec"))
    assert result.exit_code == 0, result.output
    assert "unknown account gm_account" in result.output
    md = (tmp_path / "rec" / f"{LOCAL_DATE}_transcript.md").read_text(encoding="utf-8")
    assert "gm_account: Witajcie" in md


def test_unzipped_download_missing_player_warns(tmp_path, fake_whisper, campaign):
    folder = craig_unzipped(tmp_path / "session")
    (folder / "AAAAAA_data" / "2-player_one.flac").unlink()
    result = invoke("transcribe", folder, "-c", campaign)
    assert result.exit_code == 0, result.output
    warning = "info.txt lists player_one, but no track file matches; left out"
    assert warning in result.output
    path = tmp_path / f"{LOCAL_DATE}_transcript.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert any(warning in w for w in data["warnings"])


def test_dir_input_writes_next_to_folder(tmp_path, fake_whisper, campaign):
    folder = craig_dir(tmp_path / "proc" / "tracks")
    assert invoke("transcribe", folder, "-c", campaign).exit_code == 0
    assert (tmp_path / "proc" / f"{LOCAL_DATE}_transcript.md").exists()


def test_dir_input_without_info_txt_with_start(tmp_path, fake_whisper, campaign):
    folder = craig_dir(tmp_path / "proc" / "tracks", info=False)
    result = invoke("transcribe", folder, "-c", campaign, "--start", "2026-02-03 19:30")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "proc" / "2026-02-03_transcript.md").exists()


def test_dir_input_without_info_txt_asks_for_start(tmp_path, fake_whisper, campaign):
    folder = craig_dir(tmp_path / "tracks", info=False)
    result = invoke("transcribe", folder, "-c", campaign)
    assert result.exit_code == 1 and "--start" in result.output


@pytest.mark.parametrize("text", ["2026-02-03", "03.02.2026 19:30", "2026-02-30 19:30"])
def test_bad_start_is_a_usage_error(tmp_path, fake_whisper, campaign, text):
    folder = craig_dir(tmp_path / "tracks", info=False)
    result = invoke("transcribe", folder, "-c", campaign, "--start", text)
    assert result.exit_code == 2
    assert "YYYY-MM-DD HH:MM" in result.output


def test_dir_input_missing_track(tmp_path, fake_whisper, campaign):
    folder = craig_dir(tmp_path / "tracks", skip={"player_one"})
    result = invoke("transcribe", folder, "-c", campaign)
    assert result.exit_code == 1 and "player_one" in result.output


def test_merge_rebuilds_without_touching_json(tmp_path, fake_whisper, campaign):
    zip_path = craig_zip(tmp_path / "rec")
    assert invoke("transcribe", zip_path, "-c", campaign, "--name", "s").exit_code == 0
    json_path = tmp_path / "rec" / "s_transcript.json"
    before = json_path.read_bytes()
    text = campaign.read_text(encoding="utf-8").replace("Game Master", "Narrator")
    campaign.write_text(text, encoding="utf-8")
    result = invoke("merge", json_path, "-c", campaign)
    assert result.exit_code == 0, result.output
    md = (tmp_path / "rec" / "s_transcript.md").read_text(encoding="utf-8")
    assert "Narrator: Witajcie" in md
    assert json_path.read_bytes() == before


def test_merge_keep_fillers_and_name(tmp_path, fake_whisper, campaign):
    zip_path = craig_zip(tmp_path / "rec")
    assert invoke("transcribe", zip_path, "-c", campaign, "--name", "s").exit_code == 0
    v2 = tmp_path / "v2"
    result = invoke(
        "merge", tmp_path / "rec" / "s_transcript.json", "-c", campaign,
        "--name", "v2", "-o", v2, "--keep-fillers",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (v2 / "v2_transcript.md").exists()
    assert (v2 / "v2_transcript-report.md").exists()
    assert not (v2 / "v2_transcript.json").exists()


def test_merge_keeps_the_session_name(tmp_path, fake_whisper, campaign):
    assert (
        invoke("transcribe", craig_zip(tmp_path / "rec"), "-c", campaign).exit_code == 0
    )
    rec = tmp_path / "rec"
    (rec / f"{LOCAL_DATE}_transcript.md").unlink()
    result = invoke("merge", rec / f"{LOCAL_DATE}_transcript.json", "-c", campaign)
    assert result.exit_code == 0, result.output
    assert (rec / f"{LOCAL_DATE}_transcript.md").exists()
    assert not (rec / f"{LOCAL_DATE}_transcript_transcript.md").exists()


def test_merge_json_without_suffix(tmp_path, fake_whisper, campaign):
    zip_path = craig_zip(tmp_path / "rec")
    assert invoke("transcribe", zip_path, "-c", campaign, "--name", "s").exit_code == 0
    rec = tmp_path / "rec"
    plain = rec / "plain.json"
    plain.write_bytes((rec / "s_transcript.json").read_bytes())
    result = invoke("merge", plain, "-c", campaign)
    assert result.exit_code == 0, result.output
    assert (rec / "plain_transcript.md").exists()
    assert (rec / "plain_transcript-report.md").exists()


def test_merge_unknown_format(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"format": 99}', encoding="utf-8")
    result = invoke("merge", path)
    assert result.exit_code == 1
    assert "unknown format" in result.output and "Traceback" not in result.output


def test_config_error_is_one_line(tmp_path, fake_whisper):
    bad = tmp_path / "bad.yaml"
    bad.write_text("transcribe:\n  whisper:\n    bach_size: 2\n", encoding="utf-8")
    result = invoke("transcribe", craig_zip(tmp_path / "rec"), "-c", bad)
    assert result.exit_code == 1
    assert "bad.yaml: transcribe.whisper.bach_size: unknown key" in result.output
    assert "Traceback" not in result.output


def missing_whisper(fake):
    fake.models[0] = models.Model("Whisper model", "large-v3", "huggingface", False)


def test_no_config_warns_about_defaults(tmp_path, fake_whisper):
    result = invoke("transcribe", craig_zip(tmp_path / "rec"))
    assert result.exit_code == 0, result.output
    assert "no config" in result.output and "language en" in result.output


def test_model_step_shows_model_language_and_device(tmp_path, fake_whisper, campaign):
    result = invoke("transcribe", craig_zip(tmp_path / "rec"), "-c", campaign)
    assert "load Whisper large-v3 (en) on cuda" in result.output


def test_missing_model_abort(tmp_path, fake_whisper, campaign):
    missing_whisper(fake_whisper)
    result = CliRunner().invoke(
        main, ["transcribe", str(craig_zip(tmp_path / "rec")), "-c", str(campaign)],
        input="a\n",
    )  # fmt: skip
    assert result.exit_code == 1
    assert "Whisper model large-v3 (2.9 GB) is not downloaded" in result.output
    assert "--yes" in result.output and "Traceback" not in result.output
    assert "open session.zip" not in result.output  # checked before unzipping
    assert fake_whisper.downloads == [] and fake_whisper.calls == []


def test_missing_model_download_when_asked(tmp_path, fake_whisper, campaign):
    missing_whisper(fake_whisper)
    result = CliRunner().invoke(
        main, ["transcribe", str(craig_zip(tmp_path / "rec")), "-c", str(campaign)],
        input="d\n",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert fake_whisper.downloads == ["large-v3"]


def test_missing_model_yes_downloads_without_asking(tmp_path, fake_whisper, campaign):
    missing_whisper(fake_whisper)
    result = invoke("transcribe", craig_zip(tmp_path / "rec"), "-c", campaign, "--yes")
    assert result.exit_code == 0, result.output
    assert fake_whisper.downloads == ["large-v3"]
    assert "Choose" not in result.output


def test_missing_model_use_other_downloaded_one(tmp_path, fake_whisper, campaign):
    missing_whisper(fake_whisper)
    fake_whisper.others = ["medium"]
    out = tmp_path / "out"
    result = CliRunner().invoke(
        main, ["transcribe", str(craig_zip(tmp_path / "rec")), "-c", str(campaign),
               "-o", str(out), "--name", "s"],
        input="u\n",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "use medium for this run" in result.output
    assert fake_whisper.downloads == []
    data = json.loads((out / "s_transcript.json").read_text(encoding="utf-8"))
    assert data["whisper"]["model"] == "medium"


def test_missing_alignment_model_offers_no_other_model(
    tmp_path, fake_whisper, campaign
):
    fake_whisper.models[1] = models.Model(
        "alignment model", "WAV2VEC2_ASR_BASE_960H", "torchaudio", False, damaged=True
    )
    fake_whisper.others = ["medium"]
    result = CliRunner().invoke(
        main, ["transcribe", str(craig_zip(tmp_path / "rec")), "-c", str(campaign)],
        input="d\n",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "damaged" in result.output and "[u]" not in result.output
    assert fake_whisper.downloads == ["WAV2VEC2_ASR_BASE_960H"]


def test_merge_without_config_warns(tmp_path, fake_whisper, campaign):
    assert (
        invoke("transcribe", craig_zip(tmp_path / "rec"), "--name", "s").exit_code == 0
    )
    result = invoke("merge", tmp_path / "rec" / "s_transcript.json")
    assert result.exit_code == 0, result.output
    assert "no config" in result.output


# --- finding scribe.yaml next to the recording ---------------------------------


def write_config_next_to(folder, name="Game Master"):
    (folder / "scribe.yaml").write_text(
        f"speakers:\n  gm.account: {name}\n", encoding="utf-8"
    )


def test_transcribe_uses_config_next_to_recording(tmp_path, fake_whisper):
    zip_path = craig_zip(tmp_path / "rec")
    write_config_next_to(tmp_path / "rec", "Narrator")
    result = invoke("transcribe", zip_path)
    assert result.exit_code == 0, result.output
    assert "using config scribe.yaml" in result.output
    assert "no config" not in result.output
    md = (tmp_path / "rec" / f"{LOCAL_DATE}_transcript.md").read_text(encoding="utf-8")
    assert "Narrator: Witajcie" in md


def test_transcribe_offers_to_create_config(tmp_path, fake_whisper, monkeypatch):
    import scribeio.cli as cli

    monkeypatch.setattr(cli, "_interactive", lambda: True)
    zip_path = craig_zip(tmp_path / "rec")
    result = CliRunner().invoke(main, ["transcribe", str(zip_path)], input="y\n")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "rec" / "scribe.yaml").exists()
    assert "edit" in result.output and "scribe transcribe" in result.output
    assert fake_whisper.calls == []  # stopped: nothing transcribed
    assert not (tmp_path / "rec" / f"{LOCAL_DATE}_transcript.md").exists()


def test_transcribe_declining_config_uses_defaults(tmp_path, fake_whisper, monkeypatch):
    import scribeio.cli as cli

    monkeypatch.setattr(cli, "_interactive", lambda: True)
    zip_path = craig_zip(tmp_path / "rec")
    result = CliRunner().invoke(main, ["transcribe", str(zip_path)], input="n\n")
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "rec" / "scribe.yaml").exists()
    assert fake_whisper.calls == ["gm_account", "player_one"]


def test_merge_uses_config_next_to_json(tmp_path, fake_whisper):
    zip_path = craig_zip(tmp_path / "rec")
    assert invoke("transcribe", zip_path, "--name", "s").exit_code == 0
    write_config_next_to(tmp_path / "rec", "Narrator")
    result = invoke("merge", tmp_path / "rec" / "s_transcript.json")
    assert result.exit_code == 0, result.output
    md = (tmp_path / "rec" / "s_transcript.md").read_text(encoding="utf-8")
    assert "Narrator: Witajcie" in md


def test_merge_without_config_points_to_init(tmp_path, fake_whisper):
    assert (
        invoke("transcribe", craig_zip(tmp_path / "rec"), "--name", "s").exit_code == 0
    )
    result = invoke("merge", tmp_path / "rec" / "s_transcript.json")
    assert "scribe init" in result.output


def test_transcribe_uses_cleanup_frame_ms(tmp_path, fake_whisper):
    # One 20 s frame: half tone, so the line over the silence counts as voiced
    config = tmp_path / "scribe.yaml"
    config.write_text(
        "transcribe:\n  cleanup:\n    frame_ms: 20000\n", encoding="utf-8"
    )
    zip_path = craig_zip(tmp_path / "rec")
    result = invoke("transcribe", zip_path, "-c", config, "--name", "s")
    assert result.exit_code == 0, result.output
    report = (tmp_path / "rec" / "s_transcript-report.md").read_text(encoding="utf-8")
    assert "(0% voiced)" not in report


def test_any_existing_output_needs_force(tmp_path, fake_whisper, campaign):
    (tmp_path / "rec").mkdir()
    (tmp_path / "rec" / "s_transcript-report.md").write_text("old", encoding="utf-8")
    zip_path = craig_zip(tmp_path / "rec")
    result = invoke("transcribe", zip_path, "-c", campaign, "--name", "s")
    assert result.exit_code == 1 and "--force" in result.output
    assert not (tmp_path / "rec" / "s_transcript.md").exists()


def test_glossary_option_replaces_the_config_glossary(tmp_path, fake_whisper, campaign):
    other = tmp_path / "other.txt"
    other.write_text("Karak Varn\n", encoding="utf-8")
    zip_path = craig_zip(tmp_path / "rec")
    result = invoke("transcribe", zip_path, "-c", campaign, "-g", other, "--name", "s")
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "rec" / "s_transcript.json").read_text("utf-8"))
    assert data["whisper"]["initial_prompt"] == "Karak Varn"


def test_unexpected_error_keeps_its_traceback(tmp_path, fake_whisper, campaign):
    fake_whisper.fail = RuntimeError("a bug")
    result = invoke("transcribe", craig_zip(tmp_path / "rec"), "-c", campaign)
    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)  # not turned into one line


def test_music_tracks_are_named(tmp_path, fake_whisper, campaign):
    result = invoke("transcribe", craig_zip(tmp_path / "rec"), "-c", campaign)
    assert result.exit_code == 0, result.output
    assert "music (not transcribed): Kenku_FM_1234\n" in result.output
