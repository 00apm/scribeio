"""WhisperX: checks, transcription and word alignment.

The only module that imports ``whisperx`` or ``torch``, and only inside
functions, so the rest of scribe (and ``scribe merge``) works without them.

WhisperX is used through its public API only (``load_model``,
``load_align_model``, ``align``, ``load_audio``, ``progress_callback``), so a
user's own WhisperX install works as long as it is a compatible 3.x release.
Tested with WhisperX 3.8.6; other versions get a warning, never a block.
"""

from __future__ import annotations

import gc
import logging
import math
import re
import shutil
import subprocess
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from scribe_rpg.errors import ScribeError
from scribe_rpg.transcribe.session import Segment, Word

TESTED = "3.8.6"
TESTED_MIN = (3, 8, 6)  # older versions lack APIs scribe uses
TESTED_BELOW = (4, 0, 0)  # a new major version may change the API

INSTALL_HINT = (
    'WhisperX not installed: pip install "whisperx>=3.8.6,<4" '
    "(install torch with CUDA first; see README)"
)
FFMPEG_HINT = (
    'ffmpeg not found in PATH; install it (conda install -c conda-forge "ffmpeg<8", '
    "winget install ffmpeg, apt install ffmpeg or brew install ffmpeg)"
)

Progress = Callable[[float], None]


def _whisperx() -> Any:
    """Import whisperx, or raise a ScribeError with install instructions."""
    try:
        import whisperx
    except ImportError:
        raise ScribeError(INSTALL_HINT) from None
    return whisperx


def preflight(cfg: dict[str, Any]) -> list[str]:
    """Check what `scribe transcribe` needs, before any slow work.

    Args:
        cfg: The ``transcribe.whisper`` settings.

    Returns:
        Warnings: an untested WhisperX version, or running on the CPU.

    Raises:
        ScribeError: If ffmpeg or WhisperX is missing, or the settings cannot
            work on this machine (CUDA requested but unusable; float16 on CPU).
    """
    quiet_libraries()
    if shutil.which("ffmpeg") is None:
        raise ScribeError(FFMPEG_HINT)
    _whisperx()
    import torch

    warnings = []
    found = _installed_version("whisperx")
    parsed = _parse_version(found)
    if parsed and not TESTED_MIN <= parsed < TESTED_BELOW:
        warnings.append(
            f"WhisperX {found} is outside the tested range (>=3.8.6,<4; tested "
            f"{TESTED}); transcription may fail"
        )
    if cfg["device"] != "cuda":
        if cfg["compute_type"] == "float16":
            raise ScribeError(
                "compute_type float16 does not work on the CPU; set "
                "transcribe.whisper.compute_type to int8, float32 or default"
            )
        warnings.append(
            f"Whisper runs on {cfg['device']}: transcription will be very slow "
            "(hours per track)"
        )
        return warnings
    if torch.version.cuda is None:
        raise ScribeError(
            "torch is a CPU-only build; reinstall torch with CUDA (see README), "
            "or set transcribe.whisper.device: cpu in config (very slow)"
        )
    if not torch.cuda.is_available():
        raise ScribeError(
            "CUDA not available; set transcribe.whisper.device: cpu in config to "
            "run on CPU (very slow)"
        )
    return warnings


def versions() -> dict[str, str]:
    """Versions used for this run; recorded in the JSON and the report."""
    return {
        "whisperx": _installed_version("whisperx"),
        "torch": _installed_version("torch"),
        "ffmpeg": _ffmpeg_version(),
    }


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _parse_version(text: str) -> tuple[int, int, int] | None:
    """``"3.8.6.post1"`` -> (3, 8, 6); None if it does not look like a version."""
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)


def _ffmpeg_version() -> str:
    """Version from the first line of ``ffmpeg -version``, or ``"unknown"``."""
    try:
        run = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    parts = (run.stdout or "").split()
    if parts[:2] == ["ffmpeg", "version"] and len(parts) > 2:
        return parts[2]
    return "unknown"


def load_audio(path: Path) -> Any:
    """Decode a track to 16 kHz mono float32 samples (runs ffmpeg)."""
    return _whisperx().load_audio(str(path))


def cut_prompt(
    names: Sequence[str], count_tokens: Callable[[str], int], max_tokens: int
) -> tuple[str, list[str]]:
    """Join glossary names into a Whisper prompt within a token budget.

    Args:
        names: Names in priority order (glossary file order).
        count_tokens: Token count of a text for the model's tokenizer.
        max_tokens: Budget (``transcribe.whisper.prompt_max_tokens``).

    Returns:
        The prompt (names joined with ``", "``; whole names only) and the
        names that did not fit.
    """
    used: list[str] = []
    for i, name in enumerate(names):
        if count_tokens(", ".join([*used, name])) > max_tokens:
            return ", ".join(used), list(names[i:])
        used.append(name)
    return ", ".join(used), []


def to_segments(
    aligned: list[dict[str, Any]],
    account: str,
    transcribed: Sequence[dict[str, Any]] = (),
) -> list[Segment]:
    """Convert WhisperX aligned segments into scribe segments.

    Words without a usable timestamp (missing or NaN) get times interpolated
    between their neighbours and ``estimated=True``. A segment without words
    gets one estimated word spanning it. A segment whose time cannot be
    determined at all is skipped: it could not be placed in the transcript.

    Args:
        aligned: ``whisperx.align(...)["segments"]``.
        account: Track account to stamp on every segment.
        transcribed: Segments before alignment, used to find ``avg_logprob``
            if the aligned segment lacks it.
    """
    out = []
    for a in aligned:
        text = str(a.get("text", "")).strip()
        words = [
            Word(str(w.get("word", "")).strip(), *_times(w), score=_num(w.get("score")))
            for w in a.get("words") or []
        ]
        words = [w for w in words if w.word]
        start, end = _num(a.get("start")), _num(a.get("end"))
        timed = [w for w in words if w.start is not None and w.end is not None]
        if start is None:
            start = timed[0].start if timed else None
        if end is None:
            end = timed[-1].end if timed else None
        if start is None or end is None:
            continue
        if not words:
            words = [Word(text, start, end, None, estimated=True)]
        _fill_times(words, start, end)
        logprob = _num(a.get("avg_logprob"))
        if logprob is None:
            logprob = _logprob(start, end, transcribed)
        out.append(Segment(account, start, end, text, words, avg_logprob=logprob))
    return out


def _num(v: Any) -> float | None:
    """A float rounded to ms, or None for missing / NaN values."""
    if v is None:
        return None
    f = float(v)
    return None if math.isnan(f) else round(f, 3)


def _times(w: dict[str, Any]) -> tuple[Any, Any]:
    # Word.start/end are floats once _fill_times has run; None marks "missing".
    return _num(w.get("start")), _num(w.get("end"))


def _fill_times(words: list[Word], seg_start: float, seg_end: float) -> None:
    """Give words without times an estimate, in place.

    Each run of untimed words shares the gap between the previous timed word
    (or the segment start) and the next timed word (or the segment end)
    evenly.
    """
    i = 0
    while i < len(words):
        if words[i].start is not None and words[i].end is not None:
            i += 1
            continue
        j = i
        while j < len(words) and (words[j].start is None or words[j].end is None):
            j += 1
        left = words[i - 1].end if i > 0 else seg_start
        right = words[j].start if j < len(words) else seg_end
        step = max(right - left, 0.0) / (j - i)
        for k in range(i, j):
            words[k].start = round(left + (k - i) * step, 3)
            words[k].end = round(left + (k - i + 1) * step, 3)
            words[k].estimated = True
        i = j


def _logprob(
    start: float, end: float, transcribed: Sequence[dict[str, Any]]
) -> float | None:
    """avg_logprob of the pre-alignment segment holding this segment's midpoint."""
    mid = (start + end) / 2
    for t in transcribed:
        if t.get("start", 0.0) <= mid <= t.get("end", 0.0):
            return _num(t.get("avg_logprob"))
    return None


def _call(
    func: Callable[..., Any], *args: Any, progress: Progress | None, **kwargs: Any
) -> Any:
    """Call a WhisperX function, passing `progress` as progress_callback if possible.

    WhisperX's top-level functions are ``(*args, **kwargs)`` wrappers, so the
    parameter cannot be detected from the signature. A version without it
    rejects the keyword with a TypeError when the call is bound, before any
    work starts; the call is then repeated without it.
    """
    if progress is not None:
        try:
            return func(*args, progress_callback=progress, **kwargs)
        except TypeError as e:
            if "progress_callback" not in str(e):
                raise
    return func(*args, **kwargs)


def quiet_libraries() -> None:
    """Hide info logs and notices of WhisperX and its libraries.

    They would break up scribe's progress lines; real warnings and errors
    still come through.
    """
    # WhisperX also warns per segment it cannot align; those words are
    # reported as estimated, so only its errors are shown.
    logging.getLogger("whisperx").setLevel(logging.ERROR)
    # Lightning sets its "lightning.pytorch" logger to INFO when imported, so
    # the logger printing the checkpoint-upgrade notice is set directly.
    for name in (
        "pyannote",
        "speechbrain",
        "lightning",
        "lightning_fabric",
        "pytorch_lightning",
        "lightning.pytorch.utilities.migration.utils",
        "pytorch_lightning.utilities.migration.utils",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    # pyannote turns TF32 off for reproducibility and says so on every run.
    warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")


# Messages torch gives when a model file is cut off or corrupt.
_DAMAGED = ("pytorchstreamreader", "central directory", "unexpected eof")


@contextmanager
def _oom_guard(cfg: dict[str, Any]) -> Iterator[None]:
    """Turn CUDA out-of-memory and damaged-model errors into ScribeErrors."""
    try:
        yield
    except Exception as e:
        message = str(e).lower()
        if "out of memory" in message:
            raise ScribeError(
                "CUDA out of memory; try a smaller transcribe.whisper.batch_size "
                f"(now {cfg['batch_size']})"
            ) from None
        if any(d in message for d in _DAMAGED):
            raise ScribeError(
                f"a model file is damaged ({e}); delete it from the model cache "
                "and run again"
            ) from None
        raise


# Free GPU memory below which Windows may silently move GPU memory into
# normal RAM (NVIDIA "Sysmem Fallback"), which makes transcription crawl.
LOW_VRAM = 2 * 1024**3


def _free_gpu() -> None:
    """Return memory of unloaded models to the GPU (PyTorch keeps a cache)."""
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Transcriber:
    """Whisper and alignment models, with only one of them on the GPU at a time.

    For each track Whisper transcribes, is unloaded, then the alignment model
    aligns the words and is unloaded too. Keeping both (plus PyTorch's cache)
    loaded fills an 8 GB GPU; Windows then moves memory into normal RAM
    without an error and transcription slows down several times. Reloading
    from the disk cache costs a few seconds per track.

    Attributes:
        initial_prompt: Glossary names given to Whisper.
        prompt_cut: Glossary names that did not fit the prompt.
        warnings: Problems worth showing (prompt length only estimated, little
            free GPU memory).
    """

    def __init__(self, cfg: dict[str, Any], names: list[str]) -> None:
        """Load Whisper (kept for the first track) and build the prompt.

        Args:
            cfg: The ``transcribe.whisper`` settings, passed to WhisperX as-is
                (model name or local path, device, compute type, language).
            names: Glossary names for the prompt, in priority order.

        Raises:
            ScribeError: On missing WhisperX, CUDA out of memory or a damaged
                model file.
        """
        _whisperx()
        quiet_libraries()
        self.cfg = cfg
        self.warnings: list[str] = []
        self.initial_prompt = ""
        self.prompt_cut: list[str] = []
        with _oom_guard(cfg):
            base = self._load_whisper()
            count = self._token_counter(base)
            self.initial_prompt, self.prompt_cut = cut_prompt(
                names, count, cfg["prompt_max_tokens"]
            )
            # The prompt goes in through the public asr_options; model= reuses
            # the already loaded weights (only the small VAD model is rebuilt).
            self._whisper: Any = (
                self._load_whisper(model=base.model) if self.initial_prompt else base
            )
            del base
            self._check_free_memory()

    def _load_whisper(self, **extra: Any) -> Any:
        """Load Whisper from the local cache, with the prompt once it is known.

        Models are checked (and downloaded with consent) before a run (see
        transcribe.models), so loading never touches the network.
        """
        cfg = self.cfg
        if self.initial_prompt:
            extra.setdefault("asr_options", {"initial_prompt": self.initial_prompt})
        return _whisperx().load_model(
            cfg["model"],
            cfg["device"],
            compute_type=cfg["compute_type"],
            language=cfg["language"],
            local_files_only=True,
            **extra,
        )

    def _load_alignment(self) -> tuple[Any, Any]:
        cfg = self.cfg
        try:
            model, meta = _whisperx().load_align_model(
                language_code=cfg["language"],
                device=cfg["device"],
                model_name=cfg["align_model"],
                model_cache_only=True,
            )
        except ValueError as e:
            raise ScribeError(
                f"word alignment: {e}; set transcribe.whisper.align_model to a "
                "wav2vec2 model for this language"
            ) from None
        return model, meta

    def _check_free_memory(self) -> None:
        """Warn when Whisper leaves little GPU memory for the work itself."""
        if self.cfg["device"] != "cuda":
            return
        import torch

        free, _total = torch.cuda.mem_get_info()
        if free < LOW_VRAM:
            self.warnings.append(
                f"only {free / 1024**3:.1f} GB GPU memory free after loading Whisper; "
                "Windows may then move GPU memory into RAM and transcription "
                "crawls without an error. Lower transcribe.whisper.batch_size, or "
                "set NVIDIA 'CUDA - Sysmem Fallback Policy' to 'Prefer No Sysmem "
                "Fallback' to get a clear out-of-memory error instead"
            )

    def _token_counter(self, pipeline: Any) -> Callable[[str], int]:
        """Count tokens with the model's tokenizer, or estimate if unavailable."""
        tokenizer = getattr(getattr(pipeline, "model", None), "hf_tokenizer", None)
        if tokenizer is not None:
            return lambda text: len(
                tokenizer.encode(text, add_special_tokens=False).ids
            )
        self.warnings.append(
            "could not access the Whisper tokenizer; prompt length estimated"
        )
        # Conservative estimate: about 1 token per 2 characters.
        return lambda text: math.ceil(len(text) / 2)

    def transcribe(
        self,
        audio: Any,
        account: str,
        on_progress: Progress | None = None,
        on_phase: Callable[[str], None] | None = None,
    ) -> list[Segment]:
        """Transcribe and align one track, one model on the GPU at a time.

        Args:
            audio: 16 kHz mono samples from `load_audio`.
            account: Track account stamped on the segments.
            on_progress: Called with 0..100 while the track is processed, if
                WhisperX reports progress; always called with 100 at the end.
            on_phase: Called with the current phase: loading Whisper, voice
                detection, transcribing, loading alignment model, aligning.

        Returns:
            The track's segments with word timings.
        """
        whisperx = _whisperx()
        cfg = self.cfg

        def phase(name: str) -> None:
            if on_phase is not None:
                on_phase(name)

        # WhisperX reports transcription progress only after voice detection;
        # its first report starts the "transcribing" phase. Transcription is
        # most of the work: 0-90 %; alignment: 90-100 %.
        started = False

        def transcribing(percent: float) -> None:
            nonlocal started
            if not started:
                started = True
                phase("transcribing")
            if on_progress is not None:
                on_progress(percent * 0.9)

        aligning = None
        if on_progress is not None:
            report = on_progress  # bound for the lambda (mypy narrowing)
            aligning = lambda p: report(90 + p * 0.1)  # noqa: E731

        with _oom_guard(cfg):
            if self._whisper is None:
                phase("loading Whisper")
                self._whisper = self._load_whisper()
            phase("voice detection")
            result = _call(
                self._whisper.transcribe,
                audio,
                batch_size=cfg["batch_size"],
                language=cfg["language"],
                progress=transcribing,
            )
            self._whisper = None
            _free_gpu()
            phase("loading alignment model")
            align_model, align_meta = self._load_alignment()
            phase("aligning")
            aligned = _call(
                whisperx.align,
                result["segments"],
                align_model,
                align_meta,
                audio,
                cfg["device"],
                return_char_alignments=False,
                progress=aligning,
            )
            del align_model, align_meta
            _free_gpu()
        if on_progress:
            on_progress(100.0)
        return to_segments(aligned["segments"], account, result["segments"])

    def settings(self) -> dict[str, Any]:
        """Settings used, for the JSON: config values plus the actual prompt."""
        return {
            **self.cfg,
            "initial_prompt": self.initial_prompt,
            "prompt_cut": self.prompt_cut,
        }
