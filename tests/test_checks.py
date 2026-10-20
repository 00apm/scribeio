"""Installation checks shared by `scribe check`, `transcribe` and `master`."""

import http.client
import importlib.abc
import importlib.util
import subprocess
import sys
import warnings
from email.message import Message
from types import ModuleType, SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from scribeio import checks, ffmpeg
from scribeio.checks import Item
from scribeio.errors import InstallError, ScribeError

# --- route and guide links --------------------------------------------------


def test_uv_install_detected(tmp_path):
    assert not checks.uv_install(str(tmp_path))
    (tmp_path / "uv-receipt.toml").write_text("")
    assert checks.uv_install(str(tmp_path))


def test_guide_base_is_the_public_repo():
    assert checks.GUIDE_BASE == "https://github.com/00apm/scribeio/blob/main/docs/"


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


def test_ffmpeg_path_shows_a_lowercase_extension(found_ffmpeg, monkeypatch):
    # Windows' which() returns the extension as PATHEXT spells it (.EXE)
    monkeypatch.setattr(checks.shutil, "which", lambda name: r"C:\bin\ffmpeg.EXE")
    assert checks.ffmpeg_item().detail == r"7.1.1  C:\bin\ffmpeg.exe"


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


# --- encoder ----------------------------------------------------------------


def ffmpeg_says(monkeypatch, code, stderr):
    """Make every ffmpeg run end with `code` and `stderr`; return the args seen."""
    seen = []

    def attempt(args):
        seen.append(args)
        return subprocess.CompletedProcess(args, code, "", stderr)

    monkeypatch.setattr(ffmpeg, "attempt", attempt)
    return seen


def test_encoder_ok(monkeypatch):
    seen = ffmpeg_says(monkeypatch, 0, "")
    assert checks.encoder_item("mp3", "96k") == Item("mp3 encoder", "ok")
    [args] = seen
    assert "libmp3lame" in args and "96k" in args


def test_encoder_missing(monkeypatch):
    ffmpeg_says(monkeypatch, 1, "[aost#0:0 @ 0000] Unknown encoder 'libmp3lame'\n")
    assert checks.encoder_item("mp3", "96k") == Item(
        "mp3 encoder", "missing",
        message="this ffmpeg has no libmp3lame, so it cannot write mp3 files; "
        "--format m4a works without it",
        guide="encoder",
    )  # fmt: skip


def test_encoder_missing_for_m4a_names_no_alternative(monkeypatch):
    ffmpeg_says(monkeypatch, 1, "Unknown encoder 'aac'\n")
    item = checks.encoder_item("m4a", "64k")
    assert item.status == "missing" and "--format" not in item.message


def test_encoder_refusing_bitrate_is_a_warning(monkeypatch):
    ffmpeg_says(
        monkeypatch, 1,
        "[libmp3lame @ 0x1] The bit rate 7k is unsupported\nError while opening\n",
    )  # fmt: skip
    item = checks.encoder_item("mp3", "7k")
    assert (item.status, item.detail, item.message) == (
        "warning",
        "7k",
        "The bit rate 7k is unsupported",
    )


def test_encoder_crash_without_output(monkeypatch):
    ffmpeg_says(monkeypatch, 0xC0000005, "")
    item = checks.encoder_item("mp3", "96k")
    assert (item.status, item.message) == ("warning", "ffmpeg exited with 3221225477")


@pytest.mark.ffmpeg
def test_encoder_probe_with_real_ffmpeg():
    assert checks.encoder_item("flac", "").status == "ok"


# --- WhisperX, PyTorch, GPU -------------------------------------------------


class BrokenFinder:
    """An import finder that fails like a WhisperX whose DLLs do not load."""

    def find_spec(self, name, path=None, target=None):
        if name == "whisperx":
            raise OSError("DLL load failed\nmore")
        return None


@pytest.fixture
def fake_whisperx(monkeypatch):
    # WhisperX imports lazily: torch and co. load with these two submodules
    for name in ("whisperx", "whisperx.alignment", "whisperx.asr"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace())
    monkeypatch.setattr(checks, "installed_version", lambda pkg: "3.8.6")


def test_whisperx_ok(fake_whisperx):
    assert checks.whisperx_item() == Item("WhisperX", "ok", "3.8.6")


def test_whisperx_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "whisperx", None)
    item = checks.whisperx_item()
    assert (item.status, item.message, item.guide) == (
        "missing",
        checks.WHISPERX_NOT_INSTALLED,
        "transcription",
    )


def test_whisperx_broken_install(monkeypatch):  # Review Focus 3
    monkeypatch.delitem(sys.modules, "whisperx", raising=False)
    monkeypatch.setattr(sys, "meta_path", [BrokenFinder(), *sys.meta_path])
    item = checks.whisperx_item()
    assert (item.status, item.guide) == ("missing", "transcription")
    assert item.message == "installed but does not load (OSError: DLL load failed)"


def test_whisperx_missing_dependency_is_broken(monkeypatch):
    class NoTorch:
        def find_spec(self, name, path=None, target=None):
            if name == "whisperx":
                raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.delitem(sys.modules, "whisperx", raising=False)
    monkeypatch.setattr(sys, "meta_path", [NoTorch(), *sys.meta_path])
    assert checks.whisperx_item().message.startswith(
        "installed but does not load (ModuleNotFoundError: No module named 'torch'"
    )


def test_whisperx_with_broken_torch(fake_whisperx, monkeypatch):  # Review Focus 3
    class TorchDllFails:
        def find_spec(self, name, path=None, target=None):
            if name == "whisperx.alignment":
                raise OSError("[WinError 126] c10.dll not found")

    package = ModuleType("whisperx")
    package.__path__ = []  # a package, so the finders are asked for its submodules
    monkeypatch.setitem(sys.modules, "whisperx", package)
    monkeypatch.delitem(sys.modules, "whisperx.alignment")
    monkeypatch.setattr(sys, "meta_path", [TorchDllFails(), *sys.meta_path])
    item = checks.whisperx_item()
    assert (item.status, item.guide) == ("missing", "transcription")
    assert item.message.startswith(
        "installed but does not load (OSError: [WinError 126]"
    )


class WarnsOnImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Imports `whisperx.asr` as pyannote does with ffmpeg 8+: with a warning."""

    def __init__(self, text):
        self.text = text

    def find_spec(self, name, path=None, target=None):
        if name == "whisperx.asr":
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        warnings.warn(self.text, UserWarning, stacklevel=1)


def importing_asr_warns(monkeypatch, text):
    package = ModuleType("whisperx")
    package.__path__ = []  # a package, so the finders are asked for its submodules
    monkeypatch.setitem(sys.modules, "whisperx", package)
    monkeypatch.delitem(sys.modules, "whisperx.asr")
    monkeypatch.setattr(sys, "meta_path", [WarnsOnImport(text), *sys.meta_path])


def test_whisperx_hides_the_torchcodec_warning(fake_whisperx, monkeypatch, capsys):
    # with ffmpeg 8+ pyannote warns on import; the ffmpeg item's info explains it
    importing_asr_warns(
        monkeypatch,
        "\ntorchcodec is not installed correctly so built-in audio decoding will "
        "fail. Solutions are:\n* fix torchcodec installation.",
    )
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        assert checks.whisperx_item() == Item("WhisperX", "ok", "3.8.6")
    assert shown == []
    assert "torchcodec" not in capsys.readouterr().err


def test_whisperx_lets_other_warnings_through(fake_whisperx, monkeypatch):
    importing_asr_warns(monkeypatch, "something new")
    with pytest.warns(UserWarning, match="something new"):
        checks.whisperx_item()


@pytest.mark.parametrize("found", ["3.4.2", "3.9.0", "4.0.0"])
def test_whisperx_untested_version(fake_whisperx, monkeypatch, found):
    monkeypatch.setattr(checks, "installed_version", lambda pkg: found)
    assert checks.whisperx_item() == Item(
        "WhisperX",
        "warning",
        found,
        "outside the tested range (>=3.8.6,<3.9; tested 3.8.6); transcription may fail",
    )


@pytest.mark.parametrize("found", ["3.8.7", "unknown"])
def test_whisperx_newer_3_8_or_unknown_is_ok(fake_whisperx, monkeypatch, found):
    monkeypatch.setattr(checks, "installed_version", lambda pkg: found)
    assert checks.whisperx_item().status == "ok"


# torch.cuda.get_arch_list() of PyTorch 2.8.0+cu129 on Windows
CU129_ARCHS = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
# the three warnings torch 2.8 gives for a GTX 950M (abridged)
OLD_GPU_WARNINGS = [
    "\n    Found GPU0 NVIDIA GeForce GTX 950M which is of cuda capability 5.0.\n"
    "    Minimum and Maximum cuda capability supported by this version of "
    "PyTorch is\n    (7.0) - (12.0)\n",
    "\n    Please install PyTorch with a following CUDA\n    configurations:  "
    "12.6 following instructions at\n    https://pytorch.org/get-started/locally/\n",
    "\nNVIDIA GeForce GTX 950M with CUDA capability sm_50 is not compatible "
    "with the current PyTorch installation.\n",
]


@pytest.fixture
def fake_torch(monkeypatch):
    state = SimpleNamespace(
        available=True,
        init_error=None,
        properties_error=None,
        capability=(8, 9),
        arch_list=CU129_ARCHS,
        # what torch warns on its first CUDA call (an unsupported card)
        init_warnings=[],
    )

    def init():
        if state.init_error:
            raise state.init_error

    def get_device_name(index=None):
        for text in state.init_warnings:
            warnings.warn(text, UserWarning, stacklevel=1)
        return "NVIDIA RTX"

    def get_device_properties(index):
        if state.properties_error:
            raise state.properties_error
        return SimpleNamespace(total_memory=8 * 1024**3)

    torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_available=lambda: state.available,
            init=init,
            get_device_name=get_device_name,
            get_device_properties=get_device_properties,
            get_device_capability=lambda index=None: state.capability,
            get_arch_list=lambda: state.arch_list,
        ),
    )
    state.torch = torch
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(checks, "installed_version", lambda pkg: "2.8.0")
    return state


def test_torch_cuda_build(fake_torch):
    assert checks.torch_item("cuda") == Item("PyTorch", "ok", "2.8.0, CUDA build")


def test_torch_cpu_only_build(fake_torch):
    fake_torch.torch.version.cuda = None
    assert checks.torch_item("cuda") == Item(
        "PyTorch",
        "missing",
        "2.8.0",
        "a CPU-only build, so transcription would take hours",
        "pytorch-cuda",
    )


def test_torch_cpu_device_is_info(fake_torch):
    fake_torch.torch.version.cuda = None
    assert checks.torch_item("cpu") == Item(
        "PyTorch", "info", "2.8.0", "set to run on the CPU: very slow"
    )


def test_gpu_found(fake_torch):
    item = checks.gpu_item("cuda")
    assert item == Item("GPU", "ok", message="NVIDIA RTX, 8 GB")
    assert checks.line(item) == "GPU: NVIDIA RTX, 8 GB"


def test_gpu_not_seen_gives_torch_reason(fake_torch):
    fake_torch.available = False
    fake_torch.init_error = RuntimeError("Found no NVIDIA driver\non your system")
    item = checks.gpu_item("cuda")
    assert (item.status, item.guide) == ("missing", "gpu")
    assert item.message == (
        "not seen (Found no NVIDIA driver); transcription would take hours on the CPU"
    )


def test_gpu_not_seen_with_cpu_only_torch(fake_torch):
    fake_torch.available = False
    fake_torch.torch.version.cuda = None
    assert checks.gpu_item("cuda").message.startswith(
        "not seen (PyTorch is a CPU-only build)"
    )


def test_gpu_not_seen_without_a_reason(fake_torch):
    fake_torch.available = False
    assert checks.gpu_item("cuda").message.startswith("not seen (no CUDA device found)")


def test_gpu_query_raising(fake_torch):  # Review Focus 4
    fake_torch.properties_error = RuntimeError("bad\nworse")
    item = checks.gpu_item("cuda")
    assert item.status == "missing"
    assert item.message.startswith("not seen (bad)")


def test_gpu_too_old_for_the_pytorch_build(fake_torch):
    fake_torch.capability = (5, 0)  # GTX 950M; PyTorch 2.8 cu129 starts at 7.0
    item = checks.gpu_item("cuda")
    assert item == Item(
        "GPU",
        "missing",
        message=(
            "NVIDIA RTX, 8 GB: the installed PyTorch does not support this GPU; "
            "set device: cpu to transcribe on the CPU (very slow, hours per "
            "recording)"
        ),
        guide="gpu",
    )


@pytest.mark.parametrize(
    "capability",
    [
        (7, 0),  # the oldest the build has
        (8, 9),  # between two listed (sm_86 code runs on 8.9)
        (12, 1),  # newer than all: same-major code and PTX run on it
    ],
)
def test_gpu_supported_by_the_build(fake_torch, capability):
    fake_torch.capability = capability
    assert checks.gpu_item("cuda").status == "ok"


def test_gpu_without_an_arch_list_is_not_judged(fake_torch):
    fake_torch.arch_list = []
    fake_torch.capability = (5, 0)
    assert checks.gpu_item("cuda").status == "ok"


def test_gpu_hides_torchs_old_gpu_warnings(fake_torch):
    # the ✘ line says it once; torch's three long warnings would repeat it
    fake_torch.capability = (5, 0)
    fake_torch.init_warnings = OLD_GPU_WARNINGS
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        assert checks.gpu_item("cuda").status == "missing"
    assert shown == []


def test_gpu_lets_other_warnings_through(fake_torch):
    fake_torch.init_warnings = ["something new"]
    with pytest.warns(UserWarning, match="something new"):
        checks.gpu_item("cuda")


def test_gpu_cpu_device_is_info(fake_torch):
    fake_torch.available = False
    item = checks.gpu_item("cpu")
    assert (item.status, item.message) == ("info", "set to run on the CPU: very slow")


@pytest.mark.parametrize(
    "found, shown",
    [("9.0.2", True), ("8.0", True), ("7.1.1", False), ("N-1-g", False)],
)
def test_audio_library_info(found, shown):
    item = checks.audio_library_item(found)
    assert (item is not None) == shown
    if item is not None:
        assert item.status == "info"
        assert checks.line(item) == (
            f"ffmpeg {found[0]}: WhisperX's audio library prints a warning; harmless"
        )


# --- reachability (--online) ------------------------------------------------


class Answer:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def urlopen_doing(monkeypatch, outcome):
    """checks.urlopen answers, or raises `outcome`; returns the calls."""
    calls = []

    def urlopen(request, timeout=None):
        calls.append((request, timeout))
        if outcome is not None:
            raise outcome
        return Answer()

    monkeypatch.setattr(checks, "urlopen", urlopen)
    return calls


def test_reach_ok(monkeypatch):
    calls = urlopen_doing(monkeypatch, None)
    item = checks.reach("huggingface.co", "https://huggingface.co")
    assert item == Item("huggingface.co", "ok")
    [(request, timeout)] = calls
    assert request.get_method() == "HEAD" and timeout == 10.0
    assert request.get_header("User-agent") == "scribe"


def test_reach_http_error_counts_as_reachable(monkeypatch):
    error = HTTPError("https://huggingface.co", 403, "Forbidden", Message(), None)
    urlopen_doing(monkeypatch, error)
    assert checks.reach("huggingface.co", "https://huggingface.co").status == "ok"


def test_reach_unreachable(monkeypatch):
    urlopen_doing(monkeypatch, URLError("getaddrinfo failed"))
    item = checks.reach("huggingface.co", "https://huggingface.co")
    assert (item.status, item.guide) == ("warning", "connection")
    assert item.message == "not reachable: getaddrinfo failed"


def test_reach_timeout(monkeypatch):
    calls = urlopen_doing(monkeypatch, TimeoutError())
    item = checks.reach("download.pytorch.org", checks.SOURCES["download.pytorch.org"])
    assert (item.status, item.message) == ("warning", "not reachable: timed out")
    assert calls[0][1] == 10.0


def test_sources_match_the_download_urls():
    from scribeio.transcribe import models

    assert checks.SOURCES["download.pytorch.org"] == models.TORCHAUDIO_URL


@pytest.mark.parametrize(
    "error",
    [http.client.BadStatusLine("HTTP/1.1 ???"), ValueError("bad proxy URL")],
)
def test_reach_never_fails_the_check(monkeypatch, error):
    urlopen_doing(monkeypatch, error)
    item = checks.reach("huggingface.co", "https://huggingface.co")
    assert (item.status, item.guide) == ("warning", "connection")
    assert item.message.startswith("not reachable: ")


def test_encoder_when_ffmpeg_does_not_start(monkeypatch):
    def attempt(args):
        raise OSError("[WinError 193] %1 is not a valid Win32 application")

    monkeypatch.setattr(ffmpeg, "attempt", attempt)
    item = checks.encoder_item("mp3", "96k")
    assert (item.status, item.guide) == ("missing", "ffmpeg")
    assert item.message == (
        "cannot be tried: ffmpeg does not start "
        "([WinError 193] %1 is not a valid Win32 application)"
    )
