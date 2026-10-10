"""Tests for reading Craig recordings (zip and folder of tracks)."""

import zipfile
from datetime import datetime, timezone

import pytest

from scribe_rpg.craig import is_music, open_recording, parse_info, sanitize
from scribe_rpg.errors import ScribeError

from helpers import MUSIC, TRACKS, craig_dir, craig_zip, info_txt


@pytest.mark.parametrize(
    "name, expected",
    [
        ("gm.account", "gm_account"),
        ("Kenku FM (Bot 2)#1234", "Kenku_FM__Bot_2__1234"),
        ("player-one_2", "player-one_2"),
        ("żółw.gm", "___w_gm"),
    ],
)
def test_sanitize(name, expected):
    assert sanitize(name) == expected


def test_parse_info_start_is_local_time():
    info = parse_info(info_txt())
    expected = datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc).astimezone()
    assert info.start == expected
    assert info.start.utcoffset() == expected.utcoffset()


def test_parse_info_accounts_sanitized_without_ids():
    assert parse_info(info_txt()).accounts == [
        "gm_account",
        "player_one",
        "Kenku_FM_1234",
    ]


def test_parse_info_without_start_time():
    text = info_txt().replace("Start time:", "Begin:")
    with pytest.raises(ScribeError, match="Start time"):
        parse_info(text)


def test_parse_info_bad_start_time():
    with pytest.raises(ScribeError, match="cannot read start time"):
        parse_info(info_txt(start="yesterday"))


def test_parse_info_without_tracks():
    with pytest.raises(ScribeError, match="no tracks"):
        parse_info(info_txt(tracks=[]))


def test_is_music_case_insensitive():
    assert is_music("Kenku_FM_1234", MUSIC)
    assert is_music("JUKEBOX_backup", MUSIC)
    assert not is_music("player_one", MUSIC)


def test_zip_tracks_identified(tmp_path):
    with open_recording(craig_zip(tmp_path), MUSIC) as rec:
        assert rec.source == "session.zip"
        assert [(t.account, t.kind) for t in rec.tracks] == [
            ("gm_account", "player"),
            ("player_one", "player"),
            ("Kenku_FM_1234", "music"),
        ]
        assert [t.account for t in rec.players] == ["gm_account", "player_one"]
        assert all(t.path.is_file() for t in rec.tracks)
        extracted = {p.name for p in rec.tracks[0].path.parent.parent.rglob("*")}
        assert "raw.dat" not in extracted and "AAAAAA.aup" not in extracted


def test_zip_track_order_follows_number_not_name(tmp_path):
    tracks = [TRACKS[1], TRACKS[0]]  # player_one is 1-, gm_account is 2-
    with open_recording(craig_zip(tmp_path, tracks=tracks), MUSIC) as rec:
        assert [t.account for t in rec.tracks] == ["player_one", "gm_account"]


def test_zip_temp_folder_removed(tmp_path):
    with open_recording(craig_zip(tmp_path), MUSIC) as rec:
        folder = rec.tracks[0].path.parent.parent
        assert folder.exists()
    assert not folder.exists()


def test_zip_temp_folder_removed_on_error(tmp_path):
    with pytest.raises(RuntimeError):
        with open_recording(craig_zip(tmp_path), MUSIC) as rec:
            folder = rec.tracks[0].path.parent.parent
            raise RuntimeError("boom")
    assert not folder.exists()


def test_zip_never_modified(tmp_path):
    z = craig_zip(tmp_path)
    before = z.read_bytes()
    with open_recording(z, MUSIC):
        pass
    assert z.read_bytes() == before


def test_not_a_zip(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello", encoding="utf-8")
    with pytest.raises(ScribeError, match="not a zip"):
        with open_recording(f, MUSIC):
            pass


def test_zip_without_info_txt(tmp_path):
    z = tmp_path / "other.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("AAAAAA_data/1-gm_account.flac", b"x")
    with pytest.raises(ScribeError, match="info.txt"):
        with open_recording(z, MUSIC):
            pass


def test_dir_mode(tmp_path):
    folder = craig_dir(tmp_path / "tracks")
    with open_recording(folder, MUSIC) as rec:
        assert rec.source == "tracks"
        assert [t.account for t in rec.players] == ["gm_account", "player_one"]


def test_dir_mode_music_may_be_absent(tmp_path):
    folder = craig_dir(tmp_path / "tracks", skip={"Kenku_FM_1234"})
    with open_recording(folder, MUSIC) as rec:
        assert len(rec.tracks) == 2


def test_dir_mode_missing_player_track(tmp_path):
    folder = craig_dir(tmp_path / "tracks", skip={"player_one"})
    with pytest.raises(ScribeError, match="player_one"):
        with open_recording(folder, MUSIC):
            pass


def test_dir_mode_without_info_txt(tmp_path):
    folder = craig_dir(tmp_path / "tracks")
    (folder / "info.txt").unlink()
    with pytest.raises(ScribeError, match="info.txt"):
        with open_recording(folder, MUSIC):
            pass
