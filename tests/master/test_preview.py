"""Preview files: cutting the excerpt out of every track."""

import pytest

from scribe_rpg import ffmpeg
from scribe_rpg.craig import Track
from scribe_rpg.master.preview import cut, fresh_folder

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
