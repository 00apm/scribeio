"""Tests for finding, checking and downloading the models a run needs.

Fake faster_whisper / huggingface_hub / torch / torchaudio / whisperx modules
copy the parts of their interfaces scribe uses; no network, no GPU.
"""

import io
import os
import sys
import zipfile
from types import SimpleNamespace

import pytest
from rich.console import Console

from scribeio.errors import ScribeError
from scribeio.progress import Progress
from scribeio.transcribe import models


class NotCached(FileNotFoundError):
    """Like huggingface_hub's LocalEntryNotFoundError (a FileNotFoundError)."""


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    state = SimpleNamespace(
        whisper_cached={"large-v3"},
        whisper_incomplete=set(),  # cached folder without config.json
        hf_cached={},  # repo -> folder
        hf_files={},  # repo -> {file name: size} on the Hub
        hf_error=None,  # raised by snapshot_download when downloading
        align_error=None,  # raised by load_align_model
        downloads=[],
        hub=tmp_path / "hub",
    )
    repos = {
        "medium": "Systran/faster-whisper-medium",
        "large-v3": "Systran/faster-whisper-large-v3",
        "large": "Systran/faster-whisper-large-v3",
    }

    def download_model(name, local_files_only=False, **kwargs):
        if "/" not in name and name not in repos:
            raise ValueError(f"Invalid model size '{name}'")
        if local_files_only and name not in state.whisper_cached:
            raise NotCached(name)
        # the snapshot folder, as a finished download leaves it; one in
        # whisper_incomplete lacks config.json (download cut short)
        folder = tmp_path / "snapshots" / name
        folder.mkdir(parents=True, exist_ok=True)
        for file in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
            if not (file == "config.json" and name in state.whisper_incomplete):
                (folder / file).write_text("x")
        return str(folder)

    def snapshot_download(repo, local_files_only=False, **kwargs):
        if local_files_only:
            if repo not in state.hf_cached:
                raise NotCached(repo)
            return str(state.hf_cached[repo])
        if state.hf_error:
            raise state.hf_error
        state.downloads.append(("hf", repo, kwargs.get("allow_patterns")))
        state.download_kwargs = kwargs
        return str(tmp_path / repo.replace("/", "--"))

    fw_utils = SimpleNamespace(_MODELS=repos)
    faster_whisper = SimpleNamespace(
        download_model=download_model,
        available_models=lambda: list(repos),
        utils=fw_utils,
    )
    alignment = SimpleNamespace(
        DEFAULT_ALIGN_MODELS_TORCH={"en": "WAV2VEC2_ASR_BASE_960H"},
        DEFAULT_ALIGN_MODELS_HF={"pl": "org/wav2vec2-polish"},
    )

    def load_align_model(language_code, device, model_name=None, **kwargs):
        if state.align_error:
            raise state.align_error
        state.downloads.append(("align", model_name, device))
        state.conversion_env = os.environ.get("DISABLE_SAFETENSORS_CONVERSION")
        return "model", {}

    def model_info(repo, files_metadata=False):
        files = state.hf_files[repo].items()
        return SimpleNamespace(
            siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in files]
        )

    whisperx = SimpleNamespace(alignment=alignment, load_align_model=load_align_model)
    bundle = SimpleNamespace(_path="wav2vec2_fairseq_base_ls960_asr_ls960.pth")
    torchaudio = SimpleNamespace(
        pipelines=SimpleNamespace(
            __all__=["WAV2VEC2_ASR_BASE_960H"], WAV2VEC2_ASR_BASE_960H=bundle
        )
    )
    torch = SimpleNamespace(hub=SimpleNamespace(get_dir=lambda: str(state.hub)))
    hf_constants = SimpleNamespace(HF_HUB_CACHE=str(tmp_path / "hf"))
    for name, module in {
        "faster_whisper": faster_whisper,
        "faster_whisper.utils": fw_utils,
        "huggingface_hub": SimpleNamespace(
            snapshot_download=snapshot_download,
            HfApi=lambda: SimpleNamespace(model_info=model_info),
            constants=hf_constants,
        ),
        "huggingface_hub.constants": hf_constants,
        "whisperx": whisperx,
        "whisperx.alignment": alignment,
        "torchaudio": torchaudio,
        "torch": torch,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state


def cfg(**overrides):
    c = {"model": "large-v3", "language": "en", "align_model": None}
    c.update(overrides)
    return c


def torch_file(state, valid=True):
    path = state.hub / "checkpoints" / "wav2vec2_fairseq_base_ls960_asr_ls960.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    if valid:
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("archive/data.pkl", b"x")
    else:
        path.write_bytes(b"PK\x03\x04 cut off")
    return path


# --- needed -----------------------------------------------------------------------


def test_cached_whisper_and_torchaudio_alignment(fakes):
    torch_file(fakes)
    whisper, align = models.needed(cfg())
    assert (whisper.kind, whisper.name, whisper.source, whisper.cached) == (
        "Whisper model",
        "large-v3",
        "huggingface",
        True,
    )
    assert (align.name, align.source, align.cached) == (
        "WAV2VEC2_ASR_BASE_960H",
        "torchaudio",
        True,
    )
    assert align.url.endswith("/wav2vec2_fairseq_base_ls960_asr_ls960.pth")


def test_missing_whisper_model(fakes):
    whisper, _ = models.needed(cfg(model="medium"))
    assert not whisper.cached
    assert whisper.repo == "Systran/faster-whisper-medium"


def test_unknown_whisper_model_name(fakes):
    with pytest.raises(ScribeError, match="unknown Whisper model 'huge'"):
        models.needed(cfg(model="huge"))


def test_local_model_folder_counts_as_present(fakes, tmp_path):
    folder = tmp_path / "my-model"
    folder.mkdir()
    whisper, _ = models.needed(cfg(model=str(folder)))
    assert (whisper.source, whisper.cached) == ("local", True)


def test_missing_torchaudio_alignment(fakes):
    _, align = models.needed(cfg())
    assert not align.cached and not align.damaged


def test_damaged_torchaudio_file_needs_download(fakes):
    torch_file(fakes, valid=False)
    _, align = models.needed(cfg())
    assert not align.cached and align.damaged


def test_huggingface_alignment_cached(fakes, tmp_path):
    folder = tmp_path / "pl-model"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"x")
    fakes.hf_cached["org/wav2vec2-polish"] = folder
    _, align = models.needed(cfg(language="pl"))
    assert (align.name, align.source, align.cached) == (
        "org/wav2vec2-polish",
        "huggingface",
        True,
    )


def test_huggingface_alignment_without_weights_is_missing(fakes, tmp_path):
    folder = tmp_path / "pl-model"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    fakes.hf_cached["org/wav2vec2-polish"] = folder
    _, align = models.needed(cfg(language="pl"))
    assert not align.cached


def test_align_model_setting_wins(fakes):
    _, align = models.needed(cfg(language="pl", align_model="org/other"))
    assert align.name == "org/other"


def test_language_without_alignment_model(fakes):
    with pytest.raises(ScribeError, match="transcribe.whisper.align_model"):
        models.needed(cfg(language="xx"))


@pytest.mark.parametrize("name", ["polish", "English"])
def test_language_name_instead_of_code(fakes, name):
    # not a hunt for wav2vec2 models: the value itself is wrong
    with pytest.raises(ScribeError) as err:
        models.needed(cfg(language=name))
    assert str(err.value) == (
        f"transcribe.whisper.language: {name!r} is not a language code; "
        "use a code such as en, de, pl"
    )


def test_cached_whisper_models_without_aliases(fakes):
    fakes.whisper_cached = {"medium", "large-v3", "large"}
    assert models.cached_whisper_models() == ["large-v3", "medium"]


def test_incomplete_whisper_model_is_damaged(fakes):
    # a cut-short download leaves the folder without config.json; Whisper
    # then loads and crashes in CTranslate2 mid-transcription
    fakes.whisper_cached = {"large-v3", "medium"}
    fakes.whisper_incomplete = {"medium"}
    whisper, _ = models.needed(cfg(model="medium"))
    assert (whisper.cached, whisper.damaged) == (False, True)


def test_incomplete_whisper_model_is_not_offered(fakes):
    fakes.whisper_cached = {"large-v3", "medium"}
    fakes.whisper_incomplete = {"large-v3"}
    assert models.cached_whisper_models() == ["medium"]


# --- download ------------------------------------------------------------------------


class FakeResponse(io.BytesIO):
    def __init__(self, body, length):
        super().__init__(body)
        self.headers = {"Content-Length": str(length)} if length is not None else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def quiet_progress():
    return Progress(Console(file=io.StringIO(), width=120, color_system=None))


def test_download_file_complete(monkeypatch, tmp_path):
    body = b"x" * 3000
    monkeypatch.setattr(
        models, "urlopen", lambda req, timeout: FakeResponse(body, 3000)
    )
    dst = tmp_path / "cache" / "model.pth"
    models.download_file("https://example.invalid/model.pth", dst, quiet_progress())
    assert dst.read_bytes() == body
    assert not list(dst.parent.glob("*.partial"))


def test_download_file_cut_off_is_not_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(
        models, "urlopen", lambda req, timeout: FakeResponse(b"x" * 1000, 3000)
    )
    dst = tmp_path / "model.pth"
    with pytest.raises(ScribeError, match="stopped early.*1000 of 3000") as e:
        models.download_file("https://example.invalid/model.pth", dst, quiet_progress())
    assert str(e.value).endswith(models.CONNECTION_HINT)
    assert not dst.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_download_file_network_error(monkeypatch, tmp_path):
    def fail(req, timeout):
        raise OSError("no route to host")

    monkeypatch.setattr(models, "urlopen", fail)
    with pytest.raises(ScribeError, match="download of model.pth failed") as e:
        models.download_file(
            "https://example.invalid/model.pth",
            tmp_path / "model.pth",
            quiet_progress(),
        )
    assert str(e.value).endswith(models.CONNECTION_HINT)


def test_connection_hint_names_what_to_check():
    hint = models.CONNECTION_HINT
    for part in ("internet connection", "firewall", "antivirus", "proxy", "again"):
        assert part in hint


def test_failed_whisper_download_gives_the_hint(fakes):
    fakes.hf_error = OSError("An error happened while trying to locate the files")
    whisper, _ = models.needed(cfg(model="medium"))
    with pytest.raises(ScribeError, match="download of medium failed") as e:
        models.download(whisper, quiet_progress())
    assert str(e.value).endswith(models.CONNECTION_HINT)


def test_failed_write_during_download_does_not_blame_the_connection(fakes):
    # e.g. WinError 1314 (no symlink rights) or a full disk: the error names
    # a file, which network errors do not
    fakes.hf_error = OSError(22, "A required privilege is not held", "blob", None, 1314)
    whisper, _ = models.needed(cfg(model="medium"))
    with pytest.raises(ScribeError, match="download of medium failed") as e:
        models.download(whisper, quiet_progress())
    assert "internet" not in str(e.value)
    assert str(e.value).endswith(models.DISK_HINT.format(folder=models.cache_folder()))


def test_failed_alignment_download_is_a_scribe_error(fakes):
    # WhisperX turns any loading error into this ValueError
    fakes.align_error = ValueError('The chosen align_model "x" could not be found')
    _, align = models.needed(cfg(language="pl"))
    with pytest.raises(ScribeError, match="download of org/wav2vec2-polish") as e:
        models.download(align, quiet_progress())
    assert str(e.value).endswith(models.CONNECTION_HINT)


def test_download_replaces_damaged_torchaudio_file(fakes, monkeypatch):
    path = torch_file(fakes, valid=False)
    _, align = models.needed(cfg())
    good = io.BytesIO()
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("archive/data.pkl", b"x")
    data = good.getvalue()
    monkeypatch.setattr(
        models, "urlopen", lambda req, timeout: FakeResponse(data, len(data))
    )
    models.download(align, quiet_progress())
    assert zipfile.is_zipfile(path)


def test_download_whisper_model_through_huggingface(fakes):
    whisper, _ = models.needed(cfg(model="medium"))
    models.download(whisper, quiet_progress())
    [(kind, repo, patterns)] = fakes.downloads
    assert (kind, repo) == ("hf", "Systran/faster-whisper-medium")
    assert "model.bin" in patterns


def test_whisper_download_fetches_one_file_at_a_time(fakes):
    # huggingface_hub's symlink test races with parallel downloads: on Windows
    # without symlink rights a thread then tries a symlink and fails (WinError
    # 1314), leaving the model without config.json
    whisper, _ = models.needed(cfg(model="medium"))
    models.download(whisper, quiet_progress())
    assert fakes.download_kwargs["max_workers"] == 1


def test_download_huggingface_alignment_model(fakes, monkeypatch):
    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)
    _, align = models.needed(cfg(language="pl"))
    models.download(align, quiet_progress())
    assert fakes.downloads == [("align", "org/wav2vec2-polish", "cpu")]
    # transformers would also fetch a converted copy of the weights from a
    # pull request on the Hub (1.2 GB more for Polish)
    assert fakes.conversion_env == "true"


def test_download_alignment_keeps_users_conversion_setting(fakes, monkeypatch):
    monkeypatch.setenv("DISABLE_SAFETENSORS_CONVERSION", "false")
    _, align = models.needed(cfg(language="pl"))
    models.download(align, quiet_progress())
    assert fakes.conversion_env == "false"


# --- size --------------------------------------------------------------------------

ALIGN_REPO_FILES = {
    "config.json": 2_000,
    "preprocessor_config.json": 300,
    "vocab.json": 400,
    "special_tokens_map.json": 100,
    "README.md": 5_000,
    "flax_model.msgpack": 1_261_934_272,
    "language_model/lm.binary": 1_628_858_942,
    "pytorch_model.bin": 1_262_097_815,
}


def test_size_of_alignment_model_counts_only_what_it_loads(fakes):
    fakes.hf_files["org/wav2vec2-polish"] = ALIGN_REPO_FILES
    _, align = models.needed(cfg(language="pl"))
    assert models.size(align) == 1_262_097_815 + 2_000 + 300 + 400 + 100


def test_size_of_alignment_model_prefers_safetensors(fakes):
    files = {**ALIGN_REPO_FILES, "model.safetensors": 1_261_971_432}
    fakes.hf_files["org/wav2vec2-polish"] = files
    _, align = models.needed(cfg(language="pl"))
    assert models.size(align) == 1_261_971_432 + 2_000 + 300 + 400 + 100


def test_size_of_whisper_model_counts_only_its_files(fakes):
    fakes.hf_files["Systran/faster-whisper-medium"] = {
        "model.bin": 1_500_000_000,
        "config.json": 2_000,
        "tokenizer.json": 2_000_000,
        "vocabulary.txt": 400_000,
        "README.md": 5_000,
        "model.fp32.bin": 3_000_000_000,
    }
    whisper, _ = models.needed(cfg(model="medium"))
    assert models.size(whisper) == 1_500_000_000 + 2_000 + 2_000_000 + 400_000


def test_describe():
    m = models.Model("Whisper model", "large-v3", "huggingface", False)
    assert models.describe(m, 3_100_000_000) == "Whisper model large-v3 (2.9 GB)"
    assert models.describe(m, None) == "Whisper model large-v3 (size unknown)"
    damaged = models.Model("alignment model", "X", "torchaudio", False, damaged=True)
    assert "damaged" in models.describe(damaged, 377_000_000)


# --- what is on disk --------------------------------------------------------------


def test_cache_folder(fakes):
    assert models.cache_folder() == fakes.hub.parent / "hf"


def test_local_alignment_lists_cached_and_damaged(fakes):
    torch_file(fakes)
    assert models.local_alignment() == (["en"], [])
    torch_file(fakes, valid=False)
    assert models.local_alignment() == ([], ["en"])


def test_local_alignment_huggingface_repo(fakes, tmp_path):
    folder = tmp_path / "pl-model"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"x")
    fakes.hf_cached["org/wav2vec2-polish"] = folder
    assert models.local_alignment() == (["pl"], [])


def test_nothing_on_disk_on_a_fresh_install(fakes):  # Review Focus 5
    fakes.whisper_cached = set()
    assert not fakes.hub.exists() and not models.cache_folder().exists()
    assert models.cached_whisper_models() == []
    assert models.local_alignment() == ([], [])


@pytest.mark.parametrize(
    "n, text",
    [
        (None, "size unknown"),
        (360 * 1024**2, "360 MB"),
        (int(2.9 * 1024**3), "2.9 GB"),
    ],
)
def test_amount(n, text):
    assert models.amount(n) == text
