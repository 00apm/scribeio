"""Preview files: cutting the excerpt, the per-step file list, the level."""

from pathlib import Path

import pytest

from scribe_rpg import ffmpeg
from scribe_rpg.config import load_config
from scribe_rpg.craig import Track
from scribe_rpg.master.config import resolve
from scribe_rpg.master.mix import TrackPlan
from scribe_rpg.master.preview import (
    common_level,
    cut,
    fresh_folder,
    own_chains,
    step_files,
)

from helpers import make_flac


def test_fresh_folder_replaces_old_content(tmp_path):
    folder = tmp_path / "s_preview"
    (folder / "tracks").mkdir(parents=True)
    (folder / "old.flac").write_bytes(b"x")
    assert fresh_folder(folder) == folder
    assert folder.is_dir() and not any(folder.iterdir())


def test_fresh_folder_only_for_preview_folders(tmp_path):
    with pytest.raises(ValueError):
        fresh_folder(tmp_path / "recordings")


@pytest.mark.ffmpeg
def test_cut_one_range_keeps_names(tmp_path):
    src = make_flac(tmp_path / "Gra cz'1 Łucja.flac", 10)
    [out] = cut([Track("Gra cz'1 Łucja", "player", src)], [(2, 7)], tmp_path / "t")
    assert out == tmp_path / "t" / "Gra cz'1 Łucja.flac"
    assert ffmpeg.duration(out) == pytest.approx(5, abs=0.05)


@pytest.mark.ffmpeg
def test_cut_several_ranges_joined(tmp_path):
    a = make_flac(tmp_path / "1-a.flac", 10)
    b = make_flac(tmp_path / "2-b.flac", 10, channels=2)
    tracks = [Track("a", "player", a), Track("b", "music", b)]
    outs = cut(tracks, [(1, 3), (6, 9)], tmp_path / "t")
    assert [p.name for p in outs] == ["1-a.flac", "2-b.flac"]
    for p in outs:
        assert ffmpeg.duration(p) == pytest.approx(5, abs=0.05)


@pytest.mark.ffmpeg
def test_cut_range_past_a_short_track(tmp_path):
    # a player who left early: the track ends before the range does
    short = make_flac(tmp_path / "1-a.flac", 4)
    [out] = cut([Track("a", "player", short)], [(2, 8)], tmp_path / "t")
    assert ffmpeg.duration(out) == pytest.approx(2, abs=0.05)


# --- step files (pure) --------------------------------------------------------

CFG = resolve(load_config().settings["master"])


def plans_for(tracks, cfg=CFG):
    """Full plans, as write_files builds them; gains as if measured."""
    plans = []
    for t in tracks:
        steps = cfg.chain_for(t)
        gains = {i: 0.0 for i, s in enumerate(steps) if s.type == "loudness"}
        plans.append(TrackPlan(t, steps, gains, stereo=cfg.keeps_stereo(t)))
    return plans


def full_plans(voices, music):
    tracks = [Track(f"v{i}", "player", Path(f"v{i}.flac")) for i in range(voices)]
    tracks += [Track(f"m{i}", "music", Path(f"m{i}.flac")) for i in range(music)]
    return plans_for(tracks)


def names(files):
    return [f.name for f in files]


def test_step_files_default_chain():
    files = step_files(full_plans(voices=2, music=1), CFG)
    assert names(files) == [
        "00-original.flac", "01-voice-rumble.flac", "02-voice-level.flac",
        "03-voice-comp.flac", "04-voice-eq.flac", "05-voice-loudness.flac",
        "06-music-loudness.flac", "07-mix-limiter.flac", "08-final.mp3",
    ]  # fmt: skip
    original, comp, music = files[0], files[3], files[6]
    assert [p.steps for p in original.plans] == [[], [], []]
    # voice files leave the (still untreated) music out; it joins at 06
    assert [(p.track.kind, p.upto) for p in comp.plans] == [
        ("player", 3), ("player", 3),
    ]  # fmt: skip
    assert len(original.plans) == 3
    assert comp.mix_steps == [] and comp.fmt == "flac"
    assert [p.upto for p in music.plans] == [None, None, 1]
    assert files[7].mix_steps == CFG.mix and files[7].fmt == "flac"
    assert files[8].mix_steps == CFG.mix and files[8].fmt == "mp3"
    assert all(f.channels == CFG.channels for f in files)


def test_step_files_no_voice_keeps_numbers():
    files = names(step_files(full_plans(0, 1), CFG))
    assert files == [
        "00-original.flac", "06-music-loudness.flac", "07-mix-limiter.flac",
        "08-final.mp3",
    ]  # fmt: skip


def test_step_files_no_music_keeps_numbers():
    files = names(step_files(full_plans(2, 0), CFG))
    assert "06-music-loudness.flac" not in files
    assert files[-2:] == ["07-mix-limiter.flac", "08-final.mp3"]


def test_step_files_player_with_own_chain_is_full(tmp_path):
    path = tmp_path / "scribe.yaml"
    text = "master:\n  players:\n    p1:\n      chain: [rumble, loudness]\n"
    path.write_text(text, encoding="utf-8")
    cfg = resolve(load_config(path).settings["master"])
    tracks = [Track(a, "player", Path(f"{a}.flac")) for a in ("p1", "p2")]
    plans = plans_for(tracks, cfg)
    files = step_files(plans, cfg)
    assert files[1].plans[0].upto is None and files[1].plans[1].upto == 1
    assert own_chains(plans, cfg) == ["p1"]
    assert own_chains(full_plans(2, 1), CFG) == []


def test_step_files_before_after():
    files = step_files(full_plans(2, 1), CFG, before_after=True)
    assert names(files) == ["00-original.flac", "08-final.mp3"]


@pytest.mark.parametrize(
    "levels, expected",
    [
        ([(-20.0, -10.0), (-18.0, -6.0)], -19.0),  # room enough: -19
        ([(-16.8, 3.7), (-18.4, -1.0)], -21.5),  # the cut's level file
        ([(None, None), (-20.0, -10.0)], -19.0),  # silent file ignored
    ],
)
def test_common_level(levels, expected):
    assert common_level(levels) == pytest.approx(expected)
