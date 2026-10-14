"""Tests for `scribe master` on short generated recordings."""

import re
import subprocess
from pathlib import Path

import pytest

from scribe_rpg import craig, ffmpeg
from scribe_rpg.errors import ScribeError

from helpers import (
    craig_dir,
    info_txt,
    invoke,
    make_flac,
)

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
    "extra",
    [
        ("--at", "00:01:00"),
        ("--length", "30"),
        ("--montage",),
        ("--before-after",),
        ("--solo", "gm_account"),
        ("--open",),
    ],  # fmt: skip
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
    assert "✔ measure gm_account (" in result.output
    assert "LUFS →" not in result.output  # levels only with --verbose
    assert "✔ render 2026-02-03_master.mp3 (4 tracks + mix) (" in result.output
    again = invoke("master", folder, *START_ARG)
    assert again.exit_code == 1 and "already exists" in again.output
    assert invoke("master", folder, *START_ARG, "--force").exit_code == 0


@pytest.mark.ffmpeg
def test_master_verbose_prints_levels(tmp_path):
    folder = master_dir(tmp_path / "rec" / "tracks")
    result = invoke("master", folder, *START_ARG, "--verbose")
    assert result.exit_code == 0, result.output
    level = r"level -?[\d.]+ LUFS → [+-][\d.]+ dB"
    assert re.search(rf"✔ measure gm_account: {level}, loudness ", result.output)
    assert "player_one" in result.output and "silent" in result.output


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


def render_outputs(args):
    """The files an ffmpeg render writes: the path after each ``-y``."""
    return [Path(args[i + 1]) for i, a in enumerate(args) if a == "-y"]


def fail_render(monkeypatch, written=None):
    """Make the render half-write its outputs, then fail; measuring runs.

    `written` (a list) gets the render's output paths.
    """
    real = ffmpeg.run_progress

    def fail(args, label, on_time=None):
        if not label.startswith("render"):
            return real(args, label, on_time)
        for path in render_outputs(args):
            path.write_bytes(b"half")
            if written is not None:
                written.append(path)
        raise ScribeError(f"ffmpeg failed ({label}): boom")

    monkeypatch.setattr(ffmpeg, "run_progress", fail)


@pytest.mark.ffmpeg
def test_master_failed_render_leaves_nothing(tmp_path, monkeypatch):
    folder = master_dir(tmp_path / "rec" / "tracks")
    written = []
    fail_render(monkeypatch, written)
    result = invoke("master", folder, *START_ARG)
    assert result.exit_code == 1 and "boom" in result.output
    assert written and not any(p.exists() for p in written)
    assert sorted(p.name for p in (tmp_path / "rec").iterdir()) == ["tracks"]


@pytest.mark.ffmpeg
def test_master_renders_under_partial_names(tmp_path, monkeypatch):
    # a hard stop (closed window) leaves files that say they are unfinished
    folder = tracks_dir(tmp_path)
    seen = []
    real = ffmpeg.run_progress

    def spy(args, label, on_time=None):
        if label.startswith("render"):
            seen.extend(render_outputs(args))
        return real(args, label, on_time)

    monkeypatch.setattr(ffmpeg, "run_progress", spy)
    result = invoke("master", folder, "--name", "s1", "--tracks")
    assert result.exit_code == 0, result.output
    rec = tmp_path / "rec"
    assert seen[0] == rec / "s1_master.partial.mp3"
    assert {p.parent for p in seen[1:]} == {rec / "s1_tracks.partial"}
    assert sorted(p.name for p in rec.iterdir()) == [
        "s1_master.mp3", "s1_tracks", "tracks",
    ]  # fmt: skip


@pytest.mark.ffmpeg
def test_master_force_failed_render_keeps_old_outputs(tmp_path, monkeypatch):
    folder = tracks_dir(tmp_path)
    assert invoke("master", folder, "--name", "s1", "--tracks").exit_code == 0
    rec = tmp_path / "rec"
    old = (rec / "s1_master.mp3").read_bytes()
    old_tracks = sorted(p.name for p in (rec / "s1_tracks").iterdir())
    fail_render(monkeypatch)
    result = invoke("master", folder, "--name", "s1", "--tracks", "--force")
    assert result.exit_code == 1 and "boom" in result.output
    assert (rec / "s1_master.mp3").read_bytes() == old
    assert sorted(p.name for p in (rec / "s1_tracks").iterdir()) == old_tracks
    assert sorted(p.name for p in rec.iterdir()) == [
        "s1_master.mp3", "s1_tracks", "tracks",
    ]  # fmt: skip


@pytest.mark.ffmpeg
def test_master_tracks_clears_a_leftover_partial_folder(tmp_path):
    folder = tracks_dir(tmp_path)
    leftover = tmp_path / "rec" / "s1_tracks.partial"
    leftover.mkdir()
    (leftover / "old.flac").write_bytes(b"half")
    result = invoke("master", folder, "--name", "s1", "--tracks")
    assert result.exit_code == 0, result.output
    assert not leftover.exists()
    assert not (tmp_path / "rec" / "s1_tracks" / "old.flac").exists()


def tracks_dir(tmp_path):
    """master_dir plus a Craig info.txt (start time from it, no --start)."""
    folder = master_dir(tmp_path / "rec" / "tracks")
    (folder / "info.txt").write_text(info_txt(), encoding="utf-8")
    return folder


@pytest.mark.ffmpeg
def test_master_tracks_writes_each_processed_track(tmp_path):
    folder = tracks_dir(tmp_path)
    result = invoke("master", folder, "--name", "s1", "--tracks")
    assert result.exit_code == 0, result.output
    written = tmp_path / "rec" / "s1_tracks"
    names = sorted(p.name for p in folder.iterdir())
    assert sorted(p.name for p in written.iterdir()) == names
    info = (written / "info.txt").read_text(encoding="utf-8")
    assert info == (folder / "info.txt").read_text(encoding="utf-8")
    gm = written / "1-gm_account.flac"
    assert ffmpeg.duration(gm) == pytest.approx(30, abs=0.01)
    assert "s32" in audio_stream(gm)
    assert (tmp_path / "rec" / "s1_master.mp3").exists()
    # the folder is a valid input in its own right
    with craig.open_recording(written, ["kenku"]) as rec:
        assert len(rec.tracks) == 4
    (tmp_path / "rec" / "s1_master.mp3").unlink()  # only the folder is in the way
    again = invoke("master", folder, "--name", "s1", "--tracks")
    assert again.exit_code == 1
    assert "output already exists: s1_tracks" in again.output
    forced = invoke("master", folder, "--name", "s1", "--tracks", "--force")
    assert forced.exit_code == 0, forced.output


@pytest.mark.ffmpeg
def test_master_tracks_keep_the_processing_layout(tmp_path):
    # in a stereo master a mono voice is still written mono; music stereo
    folder = tracks_dir(tmp_path)
    config = tmp_path / "scribe.yaml"
    config.write_text("master:\n  channels: stereo\n", encoding="utf-8")
    result = invoke("master", folder, "--name", "s1", "--tracks", "-c", config)
    assert result.exit_code == 0, result.output
    written = tmp_path / "rec" / "s1_tracks"
    assert ", mono," in audio_stream(written / "1-gm_account.flac")
    assert ", stereo," in audio_stream(written / "4-Kenku_FM_1234.flac")


@pytest.mark.ffmpeg
def test_master_tracks_failed_render_leaves_no_folder(tmp_path, monkeypatch):
    folder = tracks_dir(tmp_path)
    fail_render(monkeypatch)
    result = invoke("master", folder, "--name", "s1", "--tracks")
    assert result.exit_code == 1 and "boom" in result.output
    assert sorted(p.name for p in (tmp_path / "rec").iterdir()) == ["tracks"]


@pytest.mark.parametrize("extra", [("--tracks", "--force"), ("--tracks",)])
def test_master_tracks_output_is_the_input(tmp_path, extra):
    # re-mastering from a _tracks folder must not write over (delete) it
    folder = master_dir(tmp_path / "rec" / "s1_tracks")
    before = sorted(p.name for p in folder.iterdir())
    result = invoke("master", folder, *START_ARG, "--name", "s1", *extra)
    assert result.exit_code == 1
    assert "output folder s1_tracks holds the input" in result.output
    assert sorted(p.name for p in folder.iterdir()) == before


def test_master_preview_output_is_the_input(tmp_path):
    folder = master_dir(tmp_path / "rec" / "s1_preview")
    before = sorted(p.name for p in folder.iterdir())
    result = invoke("master", folder, *START_ARG, "--name", "s1", "--preview")
    assert result.exit_code == 1
    assert "output folder s1_preview holds the input" in result.output
    assert sorted(p.name for p in folder.iterdir()) == before


def test_master_tracks_not_with_preview(tmp_path):
    folder = master_dir(tmp_path / "rec" / "tracks")
    result = invoke("master", folder, "--preview", "--tracks")
    assert result.exit_code == 2
    assert "--tracks cannot be used with --preview" in result.output


STEP_FILES = [
    "00-original.flac", "01-voice-rumble.flac", "02-voice-level.flac",
    "03-voice-comp.flac", "04-voice-eq.flac", "05-voice-loudness.flac",
    "06-music-loudness.flac", "07-mix-limiter.flac", "08-final.mp3",
]  # fmt: skip


def flac_peak(path):
    """Sample peak in dBFS (astats, all channels)."""
    done = ffmpeg.run(
        ["-i", str(path), "-af", "astats=measure_perchannel=none", "-f", "null", "-"],
        "peak",
    )
    return float(re.findall(r"Peak level dB:\s*(-?[\d.]+)", done.stderr)[-1])


def run_preview(tmp_path, *extra):
    folder = preview_dir(tmp_path / "rec" / "tracks")
    args = ("master", folder, *START_ARG, "--preview", "--length", "20", *extra)
    result = invoke(*args)
    assert result.exit_code == 0, result.output
    return result, tmp_path / "rec" / "2026-02-03_preview"


def printed_level(output):
    match = re.search(r"preview level: (-?[\d.]+) LUFS", output)
    assert match is not None, output
    return float(match.group(1))


@pytest.mark.ffmpeg
def test_master_preview_writes_every_step(tmp_path):
    result, preview = run_preview(tmp_path, "--verbose")
    assert sorted(p.name for p in preview.iterdir()) == sorted([*STEP_FILES, "tracks"])
    levels = [lufs(preview / name) for name in STEP_FILES]
    assert max(levels) - min(levels) <= 0.5 and max(levels) <= -18.5
    for name in STEP_FILES[:-1]:
        assert flac_peak(preview / name) <= -0.9, name
    assert printed_level(result.output) == pytest.approx(-19.0)
    assert re.search(r"✔ measure gm_account: level -?[\d.]+ LUFS →", result.output)
    assert "✔ preview: measure 9 files (" in result.output
    assert "✔ preview: write 9 files (" in result.output
    assert not (tmp_path / "rec" / "2026-02-03_master.mp3").exists()


@pytest.mark.ffmpeg
def test_master_preview_lowers_level_for_peaks(tmp_path, monkeypatch):
    from scribe_rpg.master import preview as preview_module

    # synthetic tones peak too close to their loudness to need lowering
    monkeypatch.setattr(preview_module, "MATCH_PEAK", -30.0)
    result, preview = run_preview(tmp_path)
    level = printed_level(result.output)
    assert level < -19 and "lowered" in result.output
    levels = [lufs(preview / name) for name in STEP_FILES]
    assert max(levels) - min(levels) <= 0.5
    assert levels[0] == pytest.approx(level, abs=0.5)
    for name in STEP_FILES[:-1]:
        assert flac_peak(preview / name) <= -29.9, name


@pytest.mark.ffmpeg
def test_master_preview_before_after(tmp_path):
    _, preview = run_preview(tmp_path, "--before-after")
    names = sorted(p.name for p in preview.iterdir())
    assert names == ["00-original.flac", "08-final.mp3", "tracks"]


@pytest.mark.ffmpeg
def test_master_preview_stereo(tmp_path):
    config = tmp_path / "scribe.yaml"
    config.write_text("master:\n  channels: stereo\n", encoding="utf-8")
    _, preview = run_preview(tmp_path, "-c", config)
    for name in STEP_FILES:
        assert ", stereo," in audio_stream(preview / name), name


SOLO_FILES = [
    "00-original.flac", "01-rumble.flac", "02-level.flac", "03-comp.flac",
    "04-eq.flac", "05-loudness.flac", "06-final.mp3",
]  # fmt: skip


@pytest.mark.ffmpeg
def test_master_preview_solo(tmp_path):
    result, preview = run_preview(tmp_path, "--solo", "player_one")
    assert sorted(p.name for p in preview.iterdir()) == sorted([*SOLO_FILES, "tracks"])
    for name in SOLO_FILES:
        assert ", mono," in audio_stream(preview / name), name
    assert "✔ measure player_one (" in result.output
    assert "✔ measure gm_account" not in result.output
    # player_one speaks at 20-40 s and 60-80 s
    start = re.search(r"excerpt: (\d\d:\d\d:\d\d), 20 s", result.output)
    assert start is not None and start.group(1) in {"00:00:20", "00:01:00"}
    assert len(list((preview / "tracks").iterdir())) == 4  # all cut, for Audacity


@pytest.mark.ffmpeg
def test_master_preview_solo_matches_sanitized_names(tmp_path):
    _, preview = run_preview(tmp_path, "--solo", "Gra cz'1 Łucja")
    assert (preview / "06-final.mp3").exists()


def test_master_preview_solo_unknown_account(tmp_path):
    folder = preview_dir(tmp_path / "rec" / "tracks")
    result = invoke("master", folder, *START_ARG, "--preview", "--solo", "nobody")
    assert result.exit_code == 1
    assert "--solo: no voice track 'nobody'" in result.output
    assert "gm_account" in result.output


@pytest.mark.ffmpeg
def test_master_preview_open(tmp_path, monkeypatch):
    from scribe_rpg.master import preview as preview_module

    opened = []
    monkeypatch.setattr(preview_module, "open_folder", opened.append)
    _, preview = run_preview(tmp_path, "--before-after", "--open")
    assert opened == [preview]


@pytest.mark.ffmpeg
def test_master_preview_open_failure_is_a_warning(tmp_path, monkeypatch):
    from scribe_rpg.master import preview as preview_module

    def fail(path):
        raise ScribeError(f"could not open {path}: no file manager")

    monkeypatch.setattr(preview_module, "open_folder", fail)
    result, preview = run_preview(tmp_path, "--before-after", "--open")
    assert "⚠ could not open" in result.output
    assert (preview / "08-final.mp3").exists()
