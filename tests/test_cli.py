"""Tests for the command line, with a fake transcriber (no GPU, no audio files)."""

import json
import re
import subprocess
import textwrap
from datetime import datetime, timezone

import numpy as np
import pytest
from click.testing import CliRunner

import scribe_rpg
from scribe_rpg import ffmpeg
from scribe_rpg.cli import main
from scribe_rpg.errors import ScribeError
from scribe_rpg.transcribe import models, whisper

from helpers import craig_dir, craig_zip, make_flac, seg, silence, tone

LOCAL_DATE = datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc).astimezone()
LOCAL_DATE = LOCAL_DATE.strftime("%Y-%m-%d")
VERSIONS = {"whisperx": "3.8.6", "torch": "2.8.0+cu128", "ffmpeg": "7.1"}


class FakeTranscriber:
    """Same interface as whisper.Transcriber; returns canned segments."""

    calls: list[str] = []
    fail: Exception | None = None
    extra_warnings: list[str] = []

    def __init__(self, cfg, names):
        self.cfg = cfg
        self.initial_prompt = ", ".join(names)
        self.prompt_cut: list[str] = []
        self.warnings = list(FakeTranscriber.extra_warnings)

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
    audio = np.concatenate([tone(10.0), silence(10.0)])  # speech 0-10 s, then silence
    monkeypatch.setattr(whisper, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(whisper, "load_audio", lambda path: audio)
    monkeypatch.setattr(whisper, "preflight", lambda cfg: [])
    monkeypatch.setattr(whisper, "versions", lambda: dict(VERSIONS))
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
    config = folder / "campaign.yaml"
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


def invoke(*args):
    return CliRunner().invoke(main, [str(a) for a in args])


def test_version_option():
    result = invoke("--version")
    assert result.exit_code == 0
    assert scribe_rpg.__version__ in result.output


def test_help_texts():
    assert "Craig recording" in invoke("transcribe", "--help").output
    assert "without the GPU" in invoke("merge", "--help").output


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


def test_preflight_and_transcriber_warnings_shown(
    tmp_path, fake_whisper, campaign, monkeypatch
):
    monkeypatch.setattr(whisper, "preflight", lambda cfg: ["runs on cpu: very slow"])
    fake_whisper.extra_warnings = ["prompt length estimated"]
    result = invoke("transcribe", craig_zip(tmp_path / "rec"), "-c", campaign)
    assert result.exit_code == 0, result.output
    assert "very slow" in result.output and "prompt length estimated" in result.output


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


def test_usage_error_exit_2():
    assert invoke("transcribe").exit_code == 2


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


# --- scribe init and finding scribe.yaml ---------------------------------------


def write_config_next_to(folder, name="Game Master"):
    (folder / "scribe.yaml").write_text(
        f"speakers:\n  gm.account: {name}\n", encoding="utf-8"
    )


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
    import scribe_rpg.cli as cli

    monkeypatch.setattr(cli, "_interactive", lambda: True)
    zip_path = craig_zip(tmp_path / "rec")
    result = CliRunner().invoke(main, ["transcribe", str(zip_path)], input="y\n")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "rec" / "scribe.yaml").exists()
    assert "edit" in result.output and "scribe transcribe" in result.output
    assert fake_whisper.calls == []  # stopped: nothing transcribed
    assert not (tmp_path / "rec" / f"{LOCAL_DATE}_transcript.md").exists()


def test_transcribe_declining_config_uses_defaults(tmp_path, fake_whisper, monkeypatch):
    import scribe_rpg.cli as cli

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


def test_init_elsewhere_hints_config_option(tmp_path):
    zip_path = craig_zip(tmp_path / "rec")
    result = invoke("init", zip_path, "-o", tmp_path / "cfg")
    assert result.exit_code == 0, result.output
    assert "-c" in result.output and "scribe.yaml" in result.output.split("Next:")[1]


# --- master --preview -------------------------------------------------------

START_ARG = ("--start", "2026-02-03 19:30")


def preview_dir(folder):
    """3 voices + music, 100 s; each voice alone for 20 s, all at 60-80 s."""
    folder.mkdir(parents=True)
    make_flac(folder / "1-gm_account.flac", 100, on=[(0, 20), (60, 80)])
    make_flac(folder / "2-player_one.flac", 100, on=[(20, 40), (60, 80)])
    make_flac(folder / "Gra cz'1 Łucja.flac", 100, on=[(40, 80)])
    make_flac(folder / "4-Kenku_FM_1234.flac", 100, freq=220)
    return folder


@pytest.mark.ffmpeg
def test_master_preview_cuts_tracks(tmp_path):
    folder = preview_dir(tmp_path / "rec" / "tracks")
    result = invoke("master", folder, *START_ARG, "--preview", "--length", "20")
    assert result.exit_code == 0, result.output
    assert "excerpt: 00:01:00, 20 s" in result.output
    tracks = tmp_path / "rec" / "2026-02-03_preview" / "tracks"
    files = sorted(p.name for p in tracks.iterdir())
    assert files == [
        "1-gm_account.flac", "2-player_one.flac", "4-Kenku_FM_1234.flac",
        "Gra cz'1 Łucja.flac",
    ]  # fmt: skip
    for p in tracks.iterdir():
        assert ffmpeg.duration(p) == pytest.approx(20, abs=0.1)


@pytest.mark.ffmpeg
def test_master_preview_montage_and_fresh_folder(tmp_path):
    folder = preview_dir(tmp_path / "rec" / "tracks")
    preview = tmp_path / "rec" / "2026-02-03_preview"
    (preview / "tracks").mkdir(parents=True)
    (preview / "stale.flac").write_bytes(b"x")
    args = ("master", folder, *START_ARG, "--preview", "--length", "20")
    result = invoke(*args, "--montage")
    assert result.exit_code == 0, result.output
    # three voices: pieces of 20 // 3 = 6 s, each voice's earliest best
    assert "excerpt (montage): 00:00:00-00:00:06, 00:00:20-00:00:26, " in result.output
    assert not (preview / "stale.flac").exists()
    for p in (preview / "tracks").iterdir():
        assert ffmpeg.duration(p) == pytest.approx(18, abs=0.1)


@pytest.mark.ffmpeg
def test_master_preview_at_past_the_end(tmp_path):
    folder = preview_dir(tmp_path / "rec" / "tracks")
    result = invoke("master", folder, *START_ARG, "--preview", "--at", "99:00:00")
    assert result.exit_code == 1
    assert "past the end (00:01:40)" in result.output


@pytest.mark.parametrize(
    "extra", [("--at", "00:01:00"), ("--length", "30"), ("--montage",)]
)
def test_master_preview_options_need_preview(tmp_path, extra):
    folder = craig_dir(tmp_path / "tracks")
    result = invoke("master", folder, *extra)
    assert result.exit_code == 2 and "--preview" in result.output


def test_master_at_with_montage(tmp_path):
    folder = craig_dir(tmp_path / "tracks")
    result = invoke("master", folder, "--preview", "--at", "00:01:00", "--montage")
    assert result.exit_code == 2 and "--montage" in result.output


@pytest.mark.parametrize("text", ["1:00", "00:61:00", "ab:cd:ef"])
def test_master_bad_at(tmp_path, text):
    folder = craig_dir(tmp_path / "tracks")
    result = invoke("master", folder, "--preview", "--at", text)
    assert result.exit_code == 2 and "HH:MM:SS" in result.output


# --- master (full) ----------------------------------------------------------


def master_dir(folder):
    """Unequal lengths (longest 35 s), one silent voice, a non-ASCII name."""
    folder.mkdir(parents=True)
    make_flac(folder / "1-gm_account.flac", 30)
    make_flac(folder / "2-player_one.flac", 20, on=[])
    make_flac(folder / "Gra cz'1 Łucja.flac", 25, freq=660, volume_db=-30)
    make_flac(folder / "4-Kenku_FM_1234.flac", 35, freq=220)
    return folder


def lufs(path):
    from scribe_rpg.master.render import measure

    graph = "[0:a]ebur128=dualmono=true:framelog=verbose"
    return measure(["-i", str(path)], graph, "test")


@pytest.mark.ffmpeg
def test_master_renders_mp3(tmp_path):
    folder = master_dir(tmp_path / "rec" / "tracks")
    result = invoke("master", folder, *START_ARG)
    assert result.exit_code == 0, result.output
    out = tmp_path / "rec" / "2026-02-03_master.mp3"
    assert ffmpeg.duration(out) == pytest.approx(35, abs=0.1)
    info = ffmpeg.run(["-i", str(out), "-f", "null", "-"], "probe").stderr
    assert "mono" in info
    assert "player_one" in result.output and "silent" in result.output
    assert "gm_account: level" in result.output
    again = invoke("master", folder, *START_ARG)
    assert again.exit_code == 1 and "already exists" in again.output
    assert invoke("master", folder, *START_ARG, "--force").exit_code == 0


@pytest.mark.ffmpeg
def test_master_notes_players_not_in_recording(tmp_path):
    folder = master_dir(tmp_path / "rec" / "tracks")
    config = tmp_path / "scribe.yaml"
    config.write_text(
        "master:\n  players:\n    gm_account:\n      comp: {ratio: 4}\n"
        "    absent.one:\n      comp: {ratio: 2}\n",
        encoding="utf-8",
    )
    result = invoke("master", folder, *START_ARG, "-c", config)
    assert result.exit_code == 0, result.output
    assert "players: absent_one: not in this recording, ignored" in result.output
    assert "players: gm_account" not in result.output


@pytest.mark.ffmpeg
def test_master_checks_config_before_reading(tmp_path):
    folder = master_dir(tmp_path / "rec" / "tracks")
    config = tmp_path / "scribe.yaml"
    config.write_text("master:\n  voice:\n    comp: {ratio: 99}\n", encoding="utf-8")
    result = invoke("master", folder, *START_ARG, "-c", config)
    assert result.exit_code == 1
    assert "master.voice.comp (acompressor): Value 99" in result.output
    assert "open tracks" not in result.output


def audio_stream(path):
    """The input's ``Stream #0:0: Audio: <codec>, <rate>, <layout>`` line."""
    info = ffmpeg.run(["-i", str(path), "-f", "null", "-"], "probe").stderr
    return next(line for line in info.splitlines() if "Stream #0:0" in line)


def side_peak(path):
    """Peak dB of left minus right: about -90 (mp3 residue) when they match."""
    diff = ffmpeg.run(
        ["-i", str(path), "-af", "pan=mono|c0=c0-c1,volumedetect", "-f", "null", "-"],
        "diff",
    ).stderr
    peak = re.search(r"max_volume: (-inf|-?[\d.]+) dB", diff)
    assert peak is not None
    return float(peak.group(1))


def master_stereo(tmp_path, config_text):
    folder = master_dir(tmp_path / "rec" / "tracks")
    config = tmp_path / "scribe.yaml"
    config.write_text(config_text, encoding="utf-8")
    result = invoke("master", folder, *START_ARG, "-c", config)
    assert result.exit_code == 0, result.output
    out = tmp_path / "rec" / "2026-02-03_master.mp3"
    assert ", stereo," in audio_stream(out)
    return folder, out


@pytest.mark.ffmpeg
def test_master_stereo_mono_voices_centred(tmp_path):
    _, out = master_stereo(tmp_path, "master:\n  channels: stereo\n")
    assert side_peak(out) < -60


@pytest.mark.ffmpeg
def test_master_stereo_keeps_a_stereo_voice(tmp_path):
    folder = tmp_path / "rec" / "tracks"
    folder.mkdir(parents=True)
    # a voice recorded in stereo: different tones left and right
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=440:d=30",
         "-f", "lavfi", "-i", "sine=660:d=30", "-filter_complex",
         "[0:a][1:a]amerge=inputs=2", "-c:a", "flac",
         str(folder / "1-pro_mic.flac")],
        check=True,
    )  # fmt: skip
    make_flac(folder / "2-Kenku_FM_1234.flac", 30, freq=220)
    config = tmp_path / "scribe.yaml"
    config.write_text(
        "master:\n  channels: stereo\n  players:\n    pro_mic:\n      stereo: true\n",
        encoding="utf-8",
    )
    result = invoke("master", folder, *START_ARG, "-c", config)
    assert result.exit_code == 0, result.output
    out = tmp_path / "rec" / "2026-02-03_master.mp3"
    assert side_peak(out) > -30


@pytest.mark.ffmpeg
@pytest.mark.parametrize(("fmt", "codec"), [("m4a", "aac"), ("opus", "opus")])
def test_master_format_option(tmp_path, fmt, codec):
    folder = master_dir(tmp_path / "rec" / "tracks")
    result = invoke("master", folder, *START_ARG, "--format", fmt)
    assert result.exit_code == 0, result.output
    out = tmp_path / "rec" / f"2026-02-03_master.{fmt}"
    assert f"Audio: {codec}" in audio_stream(out)
    assert ffmpeg.duration(out) == pytest.approx(35, abs=0.1)


@pytest.mark.ffmpeg
def test_master_failed_render_leaves_nothing(tmp_path, monkeypatch):
    folder = master_dir(tmp_path / "rec" / "tracks")
    out = tmp_path / "rec" / "2026-02-03_master.mp3"

    def fail(args, label, on_time=None):
        out.write_bytes(b"half")
        raise ScribeError(f"ffmpeg failed ({label}): boom")

    monkeypatch.setattr(ffmpeg, "run_progress", fail)
    result = invoke("master", folder, *START_ARG)
    assert result.exit_code == 1 and "boom" in result.output
    assert not out.exists()


@pytest.mark.ffmpeg
def test_master_preview_writes_matched_files(tmp_path):
    folder = preview_dir(tmp_path / "rec" / "tracks")
    result = invoke("master", folder, *START_ARG, "--preview", "--length", "20")
    assert result.exit_code == 0, result.output
    preview = tmp_path / "rec" / "2026-02-03_preview"
    original, final = preview / "00-original.flac", preview / "08-final.mp3"
    assert lufs(original) == pytest.approx(-19, abs=0.5)
    assert lufs(final) == pytest.approx(-19, abs=0.5)
    assert not (tmp_path / "rec" / "2026-02-03_master.mp3").exists()


def test_cli_import_does_not_load_numpy():
    # scribe merge must work without numpy loaded (fresh interpreter: the
    # test session itself has numpy imported already)
    import subprocess
    import sys

    code = "import sys, scribe_rpg.cli; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr
