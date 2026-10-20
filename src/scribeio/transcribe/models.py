"""Find, check and download the models a transcription needs.

A run needs two models: the Whisper model (``transcribe.whisper.model``) and
a word-alignment model for the language (``transcribe.whisper.align_model``
or WhisperX's default). scribe checks the local caches first, so models the
user already has are used offline, and nothing is downloaded without the
user's consent (asked by the CLI).

Like ``whisper.py``, this module imports WhisperX and its libraries only
inside functions.
"""

from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from scribeio.errors import ScribeError
from scribeio.progress import Progress

# Files faster-whisper fetches for a model (same list as its download_model).
WHISPER_FILES = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]
# Files transformers loads for a wav2vec2 alignment model, besides one
# weights file. Repos often hold more (flax weights, a language model).
ALIGN_FILES = [
    "config.json",
    "preprocessor_config.json",
    "vocab.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "added_tokens.json",
]
# Ends a download error that came from the disk (full, no write access).
DISK_HINT = "check free disk space and that you can write to {folder}"
TORCHAUDIO_URL = "https://download.pytorch.org/torchaudio/models/"
# Ends every download error: most are a blocked or dropped connection.
CONNECTION_HINT = (
    "check your internet connection; a firewall, antivirus or proxy may be "
    "blocking Python; then run the command again"
)
# Preferred order when suggesting another downloaded Whisper model.
PREFERENCE = [
    "large-v3", "large-v2", "large-v1", "large-v3-turbo", "distil-large-v3.5",
    "distil-large-v3", "distil-large-v2", "medium", "medium.en", "small",
    "small.en", "base", "base.en", "tiny", "tiny.en",
]  # fmt: skip


@dataclass
class Model:
    """A model a run needs.

    Attributes:
        kind: ``"Whisper model"`` or ``"alignment model"``.
        name: Name as configured, e.g. ``large-v3`` or a repo id.
        source: ``"huggingface"``, ``"torchaudio"`` or ``"local"`` (a folder).
        cached: True if usable without downloading.
        damaged: True if a cached file exists but is broken (cut off).
        repo: Hugging Face repo id, if any.
        url: Download URL of a torchaudio model.
        path: Cache file of a torchaudio model.
        language: Language the alignment model is for.
    """

    kind: str
    name: str
    source: str
    cached: bool
    damaged: bool = False
    repo: str = ""
    url: str = ""
    path: Path | None = None
    language: str = ""


def needed(cfg: dict[str, Any]) -> list[Model]:
    """The Whisper model and the alignment model for these settings.

    Args:
        cfg: The ``transcribe.whisper`` settings.

    Raises:
        ScribeError: For an unknown model name, or a language without an
            alignment model.
    """
    return [_whisper(cfg["model"]), _alignment(cfg["language"], cfg["align_model"])]


def _whisper(name: str) -> Model:
    if Path(name).is_dir():
        return Model("Whisper model", name, "local", True)
    import faster_whisper
    import faster_whisper.utils

    # download_model with local_files_only only looks into the cache.
    try:
        folder = faster_whisper.download_model(name, local_files_only=True)
        complete = _whisper_complete(Path(folder))
        cached, damaged = complete, not complete
    except FileNotFoundError:  # huggingface_hub's "not in cache" error
        cached, damaged = False, False
    except ValueError as e:
        raise ScribeError(f"unknown Whisper model {name!r}: {e}") from None
    repos: dict[str, str] = getattr(faster_whisper.utils, "_MODELS", {})
    repo = name if "/" in name else repos.get(name, "")
    return Model("Whisper model", name, "huggingface", cached, damaged, repo=repo)


def _whisper_complete(folder: Path) -> bool:
    """True if a cached Whisper model has the files CTranslate2 loads.

    The cache folder exists as soon as a download starts, so a cut-short
    download looks cached; without config.json Whisper loads and then
    crashes mid-transcription.
    """
    return all((folder / f).is_file() for f in ("config.json", "model.bin")) and any(
        folder.glob("vocabulary.*")
    )


def _alignment(language: str, align_model: str | None) -> Model:
    from whisperx import alignment

    name = (
        align_model
        or alignment.DEFAULT_ALIGN_MODELS_TORCH.get(language)
        or alignment.DEFAULT_ALIGN_MODELS_HF.get(language)
    )
    if not name and not re.fullmatch(r"[a-z]{2,3}", language):
        # e.g. "polish": the value is wrong, not the alignment model
        raise ScribeError(
            f"transcribe.whisper.language: {language!r} is not a language code; "
            "use a code such as en, de, pl"
        )
    if not name:
        raise ScribeError(
            f"no word alignment model for language {language!r}; set "
            "transcribe.whisper.align_model to a wav2vec2 model for this language"
        )
    import torchaudio

    if name in torchaudio.pipelines.__all__:
        return _torchaudio_model(name, language, torchaudio)
    import huggingface_hub

    try:
        folder = Path(huggingface_hub.snapshot_download(name, local_files_only=True))
        weights = ("pytorch_model.bin", "model.safetensors")
        cached = (folder / "config.json").is_file() and any(
            (folder / w).is_file() for w in weights
        )
    except FileNotFoundError:
        cached = False
    return Model(
        "alignment model", name, "huggingface", cached, repo=name, language=language
    )


def _torchaudio_model(name: str, language: str, torchaudio: Any) -> Model:
    """A torchaudio alignment bundle and its file in torch's hub cache."""
    import torch

    file = getattr(getattr(torchaudio.pipelines, name), "_path", None)
    if not file:
        # Unknown bundle layout: let WhisperX load (and fetch) it itself.
        return Model("alignment model", name, "torchaudio", True, language=language)
    url = file if file.startswith("https") else TORCHAUDIO_URL + file
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(url).path).name
    # torch model files are zip archives: a cut-off download has no zip
    # directory, so it is caught here instead of crashing the load.
    damaged = path.is_file() and not zipfile.is_zipfile(path)
    cached = path.is_file() and not damaged
    return Model(
        "alignment model", name, "torchaudio", cached, damaged, url=url, path=path,
        language=language,
    )  # fmt: skip


def cached_whisper_models() -> list[str]:
    """Whisper models already downloaded, best first, without aliases."""
    import faster_whisper
    import faster_whisper.utils

    repos: dict[str, str] = getattr(faster_whisper.utils, "_MODELS", {})
    found: dict[str, str] = {}  # repo -> first name
    for name in faster_whisper.available_models():
        try:
            folder = faster_whisper.download_model(name, local_files_only=True)
        except (FileNotFoundError, ValueError):
            continue
        if not _whisper_complete(Path(folder)):
            continue
        found.setdefault(repos.get(name, name), name)

    def rank(name: str) -> int:
        return PREFERENCE.index(name) if name in PREFERENCE else len(PREFERENCE)

    return sorted(found.values(), key=rank)


def cache_folder() -> Path:
    """Hugging Face's model cache, where Whisper and most alignment models go."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def local_alignment() -> tuple[list[str], list[str]]:
    """Languages whose default alignment model is on disk.

    Returns:
        Sorted language codes with the model ready, and those whose
        downloaded file is damaged.
    """
    from whisperx import alignment

    languages = {
        *alignment.DEFAULT_ALIGN_MODELS_TORCH,
        *alignment.DEFAULT_ALIGN_MODELS_HF,
    }
    ready, damaged = [], []
    for language in sorted(languages):
        model = _alignment(language, None)
        if model.cached:
            ready.append(language)
        elif model.damaged:
            damaged.append(language)
    return ready, damaged


def amount(nbytes: int | None) -> str:
    """A download size for the user: ``2.9 GB``, ``360 MB`` or ``size unknown``."""
    if not nbytes:
        return "size unknown"
    if nbytes < 1024**3:
        return f"{nbytes / 1024**2:.0f} MB"
    return f"{nbytes / 1024**3:.1f} GB"


def size(model: Model) -> int | None:
    """Download size in bytes (asks the server), or None if unknown."""
    try:
        if model.source == "torchaudio" and model.url:
            request = Request(
                model.url, method="HEAD", headers={"User-Agent": "scribe"}
            )
            with urlopen(request, timeout=10) as response:
                return int(response.headers.get("Content-Length") or 0) or None
        if model.source == "huggingface" and model.repo:
            import huggingface_hub

            info = huggingface_hub.HfApi().model_info(model.repo, files_metadata=True)
            files = {f.rfilename: f.size or 0 for f in info.siblings or []}
            patterns = _patterns(model, files)
            used = [n for n in files if any(fnmatch(n, p) for p in patterns)]
            return sum(files[n] for n in used) or None
    except Exception:  # size is only informative; never block on it
        return None
    return None


def _patterns(model: Model, files: dict[str, int]) -> list[str]:
    """The repo files a model loads: Whisper's list, or alignment files plus
    the one weights file transformers picks (safetensors over .bin)."""
    if model.kind == "Whisper model":
        return WHISPER_FILES
    weights = (
        "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    )
    return [*ALIGN_FILES, weights]


def label(model: Model) -> str:
    """Short name for the user, e.g. ``Whisper model large-v3``."""
    return f"{model.kind} {model.name}"


def describe(model: Model, nbytes: int | None) -> str:
    """One line for the user, e.g. ``Whisper model large-v3 (2.9 GB)``."""
    note = ", file damaged, will be replaced" if model.damaged else ""
    return f"{label(model)} ({amount(nbytes)}{note})"


def download(model: Model, progress: Progress) -> None:
    """Download a missing model into its usual cache.

    Raises:
        ScribeError: If the download fails or stops early.
    """
    if model.source == "torchaudio" and model.url and model.path:
        download_file(model.url, model.path, progress)
        return
    progress.console.print(f"downloading {label(model)} ...", markup=False)
    try:
        if model.kind == "Whisper model":
            _download_whisper(model)
        else:
            import whisperx

            # Without this, transformers fetches a repo's .bin weights and
            # then, in the background, a converted .safetensors copy from a
            # pull request on the Hub: the same model twice (+1.2 GB for pl).
            os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")
            # Loading on the CPU fetches exactly the files WhisperX needs.
            whisperx.load_align_model(
                language_code=model.language, device="cpu", model_name=model.repo
            )
    except (OSError, ValueError) as e:
        # WhisperX reports any failure to load an alignment model, a blocked
        # download included, as a ValueError ("could not be found").
        # A disk or permission error names the file it failed on; network
        # errors do not.
        local = isinstance(e, OSError) and e.filename is not None
        hint = DISK_HINT.format(folder=cache_folder()) if local else CONNECTION_HINT
        raise ScribeError(f"download of {model.name} failed: {e}; {hint}") from None


def _download_whisper(model: Model) -> None:
    if model.repo:
        import huggingface_hub

        # Unlike faster-whisper's own download, this shows a progress bar.
        # One file at a time: huggingface_hub marks a folder "symlinks work"
        # before testing it, so with parallel downloads another thread can
        # try a symlink and fail on Windows without symlink rights (WinError
        # 1314), leaving the model without config.json.
        huggingface_hub.snapshot_download(
            model.repo, allow_patterns=WHISPER_FILES, max_workers=1
        )
        return
    import faster_whisper

    faster_whisper.download_model(model.name)


def download_file(url: str, dst: Path, progress: Progress) -> None:
    """Download `url` to `dst`, saving it only if every byte arrived.

    The file is written next to `dst` with a ``.partial`` suffix and renamed
    at the end, after its length matched the server's Content-Length.

    Raises:
        ScribeError: If the connection fails or the download stops early.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    request = Request(url, headers={"User-Agent": "scribe"})
    try:
        with urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0) or None
            received = 0
            with progress.download(dst.name, total) as advance, tmp.open("wb") as f:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
                    received += len(chunk)
                    advance(len(chunk))
                if total is not None and received != total:
                    raise ScribeError(
                        f"download of {dst.name} stopped early ({received} of "
                        f"{total} bytes); {CONNECTION_HINT}"
                    )
        tmp.replace(dst)
    except OSError as e:
        raise ScribeError(
            f"download of {dst.name} failed: {e}; {CONNECTION_HINT}"
        ) from None
    finally:
        tmp.unlink(missing_ok=True)
