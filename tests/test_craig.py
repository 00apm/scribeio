"""Tests for reading Craig recordings (zip and folder of tracks)."""

import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scribe_rpg.craig import (
    account_of,
    dir_tracks,
    is_music,
    open_recording,
    parse_info,
    read_info,
    sanitize,
    speaker_name,
)
from scribe_rpg.errors import ScribeError

from helpers import MUSIC, TRACKS, craig_dir, craig_unzipped, craig_zip, info_txt

GIVEN = datetime(2026, 2, 3, 19, 30).astimezone()


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


def test_flac_zip_has_tracks_at_the_top(tmp_path):
    # Craig's "FLAC" download: no *_data folder, no .aup
    with open_recording(craig_zip(tmp_path, layout="flac"), MUSIC) as rec:
        assert [(t.account, t.kind) for t in rec.tracks] == [
            ("gm_account", "player"),
            ("player_one", "player"),
            ("Kenku_FM_1234", "music"),
        ]
        assert rec.start == datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc)
        assert all(t.path.is_file() for t in rec.tracks)


def test_zip_without_flac_tracks_names_the_formats(tmp_path):
    with pytest.raises(ScribeError) as err:
        with open_recording(craig_zip(tmp_path, layout="aac"), MUSIC):
            pass
    assert str(err.value) == (
        "session.zip: no FLAC tracks; download the recording from Craig as "
        "FLAC or as an Audacity project"
    )


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
    with pytest.raises(ScribeError) as err:
        with open_recording(folder, MUSIC):
            pass
    # renamed tracks next to info.txt land here too: the message says how out
    assert str(err.value) == (
        "tracks: info.txt lists player_one, but no track file matches; keep "
        "Craig's track names, or remove info.txt and give --start"
    )


def test_dir_mode_without_info_txt_needs_start(tmp_path):
    folder = craig_dir(tmp_path / "tracks", info=False)
    with pytest.raises(ScribeError) as err:
        with open_recording(folder, MUSIC):
            pass
    assert str(err.value) == (
        'tracks: no info.txt; give the session start with --start "YYYY-MM-DD HH:MM"'
    )


def test_dir_mode_without_info_txt_with_start(tmp_path):
    folder = craig_dir(tmp_path / "tracks", info=False)
    with open_recording(folder, MUSIC, start=GIVEN) as rec:
        assert rec.start == GIVEN
        assert rec.info_txt is None
        assert [t.account for t in rec.players] == ["gm_account", "player_one"]


def test_dir_mode_keeps_info_txt_text(tmp_path):
    with open_recording(craig_dir(tmp_path / "tracks"), MUSIC) as rec:
        assert rec.info_txt is not None and "Tracks:" in rec.info_txt


def test_unzipped_craig_folder_reads_its_data_folder(tmp_path):
    folder = craig_unzipped(tmp_path / "session")
    with open_recording(folder, MUSIC) as rec:
        assert rec.source == "session"
        assert [(t.account, t.kind) for t in rec.tracks] == [
            ("gm_account", "player"),
            ("player_one", "player"),
            ("Kenku_FM_1234", "music"),
        ]
        assert rec.start == datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc)
        assert rec.info_txt is not None
        assert all(t.path.parent.name == "AAAAAA_data" for t in rec.tracks)


def test_unzipped_craig_folder_may_miss_music(tmp_path):
    folder = craig_unzipped(tmp_path / "session")
    (folder / "AAAAAA_data" / "3-Kenku_FM_1234.flac").unlink()
    with open_recording(folder, MUSIC) as rec:
        assert len(rec.tracks) == 2


def test_unzipped_download_without_flac_names_the_formats(tmp_path):
    folder = tmp_path / "session"
    with zipfile.ZipFile(craig_zip(tmp_path / "zip", layout="aac")) as zf:
        zf.extractall(folder)
    with pytest.raises(ScribeError) as err:
        with open_recording(folder, MUSIC):
            pass
    assert str(err.value) == (
        "session: no FLAC tracks; download the recording from Craig as FLAC "
        "or as an Audacity project"
    )


def test_two_data_folders_are_not_guessed(tmp_path):
    folder = craig_unzipped(tmp_path / "session")
    (folder / "BBBBBB_data").mkdir()
    (folder / "BBBBBB_data" / "1-x.flac").write_bytes(b"fLaC fake")
    with pytest.raises(ScribeError, match="^session: no FLAC files"):
        with open_recording(folder, MUSIC):
            pass


def test_dir_mode_no_flac(tmp_path):
    folder = craig_dir(tmp_path / "tracks", names=[])
    with pytest.raises(ScribeError, match="^tracks: no FLAC files$"):
        with open_recording(folder, MUSIC, start=GIVEN):
            pass


@pytest.mark.parametrize("make", ["zip", "dir"])
def test_given_start_wins_over_info_txt(tmp_path, make):
    path = craig_zip(tmp_path) if make == "zip" else craig_dir(tmp_path / "t")
    with open_recording(path, MUSIC, start=GIVEN) as rec:
        assert rec.start == GIVEN


@pytest.mark.parametrize(
    ("name", "account"),
    [("3-gm_account.flac", "gm_account"), ("Game Master.flac", "Game Master"),
     ("12-a-b.flac", "a-b")],
)  # fmt: skip
def test_account_of(name, account):
    assert account_of(Path(name)) == account


def test_dir_tracks_order(tmp_path):
    names = ["Zed.flac", "10-c.flac", "2-b.flac", "1-a.flac", "notes.txt"]
    folder = craig_dir(tmp_path / "t", info=False, names=names)
    assert [f.name for f in dir_tracks(folder)] == [
        "1-a.flac", "2-b.flac", "10-c.flac", "Zed.flac",
    ]  # fmt: skip


def test_dir_mode_stems_with_spaces(tmp_path):
    names = ["Game Master.flac", "Kenku FM.flac"]
    folder = craig_dir(tmp_path / "t", info=False, names=names)
    with open_recording(folder, ["Kenku FM"], start=GIVEN) as rec:
        assert [(t.account, t.kind) for t in rec.tracks] == [
            ("Game Master", "player"), ("Kenku FM", "music"),
        ]  # fmt: skip


def test_dir_mode_info_txt_matches_sanitized_names(tmp_path):
    tracks = [("Game Master#0 (000000000000000010)", "x")]
    names = ["Game Master.flac"]
    folder = craig_dir(tmp_path / "t", tracks=tracks, names=names)
    with open_recording(folder, MUSIC) as rec:
        assert [t.account for t in rec.players] == ["Game Master"]


def test_speaker_name_compares_sanitized():
    speakers = {"Game_Master": "GM"}
    assert speaker_name(speakers, "Game Master") == "GM"
    assert speaker_name(speakers, "player one") == "player one"


def test_parse_info_keeps_names_as_discord_shows_them():
    names = parse_info(info_txt()).names
    assert names == ["gm.account", "player.one", "Kenku FM#1234"]


def test_read_info_from_zip_without_extracting(tmp_path):
    info = read_info(craig_zip(tmp_path))
    assert info.accounts == ["gm_account", "player_one", "Kenku_FM_1234"]
    assert info.names[0] == "gm.account"
    assert not any(tmp_path.rglob("*_data"))  # nothing unpacked next to the zip


def test_read_info_from_folder(tmp_path):
    assert read_info(craig_dir(tmp_path / "tracks")).names[1] == "player.one"


def test_read_info_not_a_recording(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ScribeError, match="not a zip"):
        read_info(f)
