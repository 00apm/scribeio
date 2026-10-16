"""Installation checks shared by `scribe check`, `transcribe` and `master`."""

import pytest

from scribe_rpg import checks, ffmpeg
from scribe_rpg.checks import Item
from scribe_rpg.errors import InstallError, ScribeError

# --- route and guide links --------------------------------------------------


def test_uv_install_detected(tmp_path):
    assert not checks.uv_install(str(tmp_path))
    (tmp_path / "uv-receipt.toml").write_text("")
    assert checks.uv_install(str(tmp_path))


def test_guide_url_follows_the_route(monkeypatch, tmp_path):
    monkeypatch.setattr(checks.sys, "prefix", str(tmp_path))
    assert checks.guide_url("gpu") == checks.GUIDE_BASE + "install-own-env.md#gpu"
    (tmp_path / "uv-receipt.toml").write_text("")
    assert checks.guide_url("encoder") == checks.GUIDE_BASE + "install.md#mp3-encoder"


def test_every_guide_key_has_an_anchor():
    assert set(checks.ANCHORS) == {
        "ffmpeg",
        "encoder",
        "transcription",
        "pytorch-cuda",
        "gpu",
        "connection",
    }


# --- lines and errors -------------------------------------------------------


def test_line_joins_name_detail_message():
    assert checks.line(Item("PyTorch", "ok", "2.8.0, CUDA build")) == (
        "PyTorch 2.8.0, CUDA build"
    )
    assert checks.line(Item("ffmpeg", "missing", message="m")) == "ffmpeg: m"
    assert checks.line(Item("ffmpeg", "warning", "5.1", "old")) == "ffmpeg 5.1: old"


def test_require_raises_install_error_with_link():
    item = Item("ffmpeg", "missing", message="m", guide="ffmpeg")
    with pytest.raises(InstallError) as e:
        checks.require(item)
    assert str(e.value).splitlines() == [
        "ffmpeg: m",
        "see " + checks.guide_url("ffmpeg"),
    ]
    assert isinstance(e.value, ScribeError)


@pytest.mark.parametrize("status", ["ok", "warning", "info"])
def test_require_passes_other_items_through(status):
    item = Item("ffmpeg", status)
    assert checks.require(item) is item


# --- versions ---------------------------------------------------------------


def test_installed_version_of_a_missing_package():
    assert checks.installed_version("no-such-package-scribe") == "unknown"


@pytest.mark.parametrize(
    "text, parsed",
    [("3.8.6", (3, 8, 6)), ("3.8.6.post1", (3, 8, 6)), ("6.1", (6, 1, 0)),
     ("N-11234-gabc", None), ("unknown", None)],
)  # fmt: skip
def test_parse_version(text, parsed):
    assert checks.parse_version(text) == parsed


# --- ffmpeg -----------------------------------------------------------------


@pytest.fixture
def found_ffmpeg(monkeypatch):
    """ffmpeg in PATH at /bin/ffmpeg; set `.version` to choose its version."""
    state = type("State", (), {"version": "7.1.1"})()
    monkeypatch.setattr(checks.shutil, "which", lambda name: "/bin/ffmpeg")
    monkeypatch.setattr(ffmpeg, "version", lambda: state.version)
    return state


def test_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr(checks.shutil, "which", lambda name: None)
    item = checks.ffmpeg_item()
    assert (item.status, item.guide) == ("missing", "ffmpeg")
    assert item.message == "not found in PATH: scribe needs it to read and write audio"


def test_ffmpeg_found(found_ffmpeg):
    assert checks.ffmpeg_item() == Item("ffmpeg", "ok", "7.1.1  /bin/ffmpeg")


@pytest.mark.parametrize("found", ["4.4.2", "5.1.2", "6.0"])
def test_ffmpeg_old_is_a_warning(found_ffmpeg, found):
    found_ffmpeg.version = found
    item = checks.ffmpeg_item()
    assert (item.status, item.detail) == ("warning", found)
    assert item.message == (
        "older than 6.1, the oldest scribe supports; mastering may fail"
    )


@pytest.mark.parametrize("found", ["N-11234-gabc", "unknown", "6.1", "9.0.2"])
def test_ffmpeg_unparsable_or_new_is_ok(found_ffmpeg, found):
    # git builds and an unreadable version are not warned about
    found_ffmpeg.version = found
    item = checks.ffmpeg_item()
    assert item.status == "ok" and item.detail.startswith(found)
