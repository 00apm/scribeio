"""Tests for the WhisperX wrapper, with fake whisperx/torch modules (no GPU).

The fakes copy the interface of WhisperX 3.8.6 (see _dev notes, 2026-10-10).
"""

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from scribe_rpg import checks, ffmpeg
from scribe_rpg.config import load_config
from scribe_rpg.errors import InstallError, ScribeError
from scribe_rpg.transcribe import whisper


def words_count(text):
    return len(text.split())


# --- cut_prompt -----------------------------------------------------------------


def test_cut_prompt_keeps_whole_names_in_order():
    names = ["Vorgrad", "Ostromir", "Czarna Wieża"]
    prompt, cut = whisper.cut_prompt(names, words_count, 3)
    assert prompt == "Vorgrad, Ostromir"
    assert cut == ["Czarna Wieża"]


def test_cut_prompt_everything_fits():
    assert whisper.cut_prompt(["Vorgrad"], words_count, 10) == ("Vorgrad", [])


def test_cut_prompt_no_names():
    assert whisper.cut_prompt([], words_count, 10) == ("", [])


# --- to_segments ------------------------------------------------------------------


def test_to_segments_basic():
    aligned = [
        {
            "start": 1.0,
            "end": 2.0,
            "text": " To jest test",
            "avg_logprob": -0.25,
            "words": [
                {"word": "To", "start": 1.0, "end": 1.2, "score": 0.91},
                {"word": "jest", "start": 1.3, "end": 1.5, "score": 0.8},
                {"word": "test", "start": 1.6, "end": 2.0, "score": 0.7},
            ],
        }
    ]
    [s] = whisper.to_segments(aligned, "gm_account")
    assert (s.account, s.start, s.end, s.text, s.avg_logprob) == (
        "gm_account",
        1.0,
        2.0,
        "To jest test",
        -0.25,
    )
    assert [w.word for w in s.words] == ["To", "jest", "test"]
    assert not any(w.estimated for w in s.words)


def test_avg_logprob_falls_back_to_transcribed_segment():
    aligned = [{"start": 1.0, "end": 2.0, "text": "Tak", "words": []}]
    transcribed = [{"start": 0.5, "end": 2.5, "text": "Tak", "avg_logprob": -0.4}]
    [s] = whisper.to_segments(aligned, "gm_account", transcribed)
    assert s.avg_logprob == -0.4


def test_missing_timestamps_interpolated():
    aligned = [
        {
            "start": 0.0,
            "end": 4.0,
            "text": "w roku 1923 i 1924 tak",
            "words": [
                {"word": "w", "start": 0.0, "end": 0.5},
                {"word": "roku", "start": 0.5, "end": 1.0},
                {"word": "1923"},
                {"word": "i"},
                {"word": "1924", "start": 3.0, "end": 3.5},
                {"word": "tak"},
            ],
        }
    ]
    [s] = whisper.to_segments(aligned, "gm_account")
    times = [(w.word, w.start, w.end, w.estimated) for w in s.words]
    assert times[2] == ("1923", 1.0, 2.0, True)
    assert times[3] == ("i", 2.0, 3.0, True)
    assert times[5] == ("tak", 3.5, 4.0, True)  # up to segment end
    assert times[0][3] is False


def test_nan_timestamps_count_as_missing():
    aligned = [
        {
            "start": 2.0,
            "end": 3.0,
            "text": "12 lat",
            "words": [
                {"word": "12", "start": math.nan, "end": math.nan, "score": math.nan},
                {"word": "lat", "start": 2.5, "end": 3.0},
            ],
        }
    ]
    [s] = whisper.to_segments(aligned, "gm_account")
    w = s.words[0]
    assert (w.start, w.end, w.score, w.estimated) == (2.0, 2.5, None, True)


def test_segment_without_words_gets_one_estimated_word():
    aligned = [{"start": 5.0, "end": 6.0, "text": " 1923.", "words": []}]
    [s] = whisper.to_segments(aligned, "gm_account")
    assert [(w.word, w.start, w.end, w.estimated) for w in s.words] == [
        ("1923.", 5.0, 6.0, True)
    ]


def test_segment_times_from_words_when_missing():
    aligned = [
        {
            "start": math.nan,
            "end": None,
            "text": "Tak",
            "words": [{"word": "Tak", "start": 7.0, "end": 7.4}],
        }
    ]
    [s] = whisper.to_segments(aligned, "gm_account")
    assert (s.start, s.end) == (7.0, 7.4)


def test_segment_without_any_time_is_skipped():
    aligned = [{"start": math.nan, "end": math.nan, "text": "Tak", "words": []}]
    assert whisper.to_segments(aligned, "gm_account") == []


# --- fake whisperx / torch (WhisperX 3.8.6 interface) -----------------------------


@dataclass
class FakeState:
    cuda: bool = True
    cuda_build: str | None = "12.8"
    whisperx_version: str = "3.8.6"
    load_error: Exception | None = None
    align_error: Exception | None = None
    tokenizer: bool = True
    progress: bool = True
    loads: list = field(default_factory=list)
    align_loads: list = field(default_factory=list)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=text.split())


class FakePipeline:
    def __init__(self, state, asr_options, model):
        self.state = state
        self.asr_options = asr_options or {}
        tokenizer = FakeTokenizer() if state.tokenizer else None
        self.model = model or SimpleNamespace(hf_tokenizer=tokenizer)

    def transcribe(self, audio, batch_size=None, language=None, progress_callback=None):
        if progress_callback:
            progress_callback(50.0)
            progress_callback(100.0)
        return {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": " To jest test", "avg_logprob": -0.2}
            ],
            "language": language,
        }


class FakePipelineNoProgress(FakePipeline):
    """A WhisperX without the progress_callback parameter."""

    def transcribe(self, audio, batch_size=None, language=None):
        return super().transcribe(audio, batch_size, language)


def fake_align(*args, **kwargs):
    """Like whisperx.align: a (*args, **kwargs) wrapper around the real function."""
    return _fake_align(*args, **kwargs)


def _fake_align(
    segments,
    model,
    meta,
    audio,
    device,
    return_char_alignments=False,
    progress_callback=None,
):
    if progress_callback:
        progress_callback(100.0)
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": " To jest test",
                "avg_logprob": -0.2,
                "words": [
                    {"word": "To", "start": 0.0, "end": 0.2, "score": 0.9},
                    {"word": "jest", "start": 0.3, "end": 0.5, "score": 0.9},
                    {"word": "test", "start": 0.6, "end": 1.0, "score": 0.9},
                ],
            }
        ]
    }


@pytest.fixture
def fakes(monkeypatch):
    state = FakeState()

    def load_model(
        arch,
        device,
        compute_type="default",
        asr_options=None,
        language=None,
        model=None,
        **kwargs,
    ):
        if state.load_error:
            raise state.load_error
        state.loads.append(
            {
                "arch": arch,
                "device": device,
                "compute_type": compute_type,
                "language": language,
                "asr_options": asr_options,
                "model": model,
                "local_files_only": kwargs.get("local_files_only"),
            }
        )
        cls = FakePipeline if state.progress else FakePipelineNoProgress
        return cls(state, asr_options, model)

    def load_align_model(language_code, device, model_name=None, **kwargs):
        if state.align_error:
            raise state.align_error
        state.align_loads.append((language_code, device, model_name))
        state.align_cache_only = kwargs.get("model_cache_only")
        return "align-model", {"language": language_code}

    fake_whisperx = SimpleNamespace(
        load_model=load_model, load_align_model=load_align_model, align=fake_align
    )
    state.freed = 0
    state.free_bytes = 6 * 1024**3

    def empty_cache():
        state.freed += 1

    def init():
        if not state.cuda:
            raise RuntimeError("Found no NVIDIA driver on your system")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: state.cuda,
            init=init,
            get_device_name=lambda index=None: "NVIDIA RTX",
            get_device_properties=lambda index: SimpleNamespace(
                total_memory=8 * 1024**3
            ),
            empty_cache=empty_cache,
            mem_get_info=lambda: (state.free_bytes, 8 * 1024**3),
        ),
        version=SimpleNamespace(cuda=state.cuda_build),
    )
    state.torch = fake_torch
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(checks.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(
        checks,
        "installed_version",
        lambda pkg: {"whisperx": state.whisperx_version, "torch": "2.8.0"}[pkg],
    )
    return state


def wcfg(**overrides):
    cfg = load_config().settings["transcribe"]["whisper"]
    cfg.update(overrides)
    return cfg


# --- preflight --------------------------------------------------------------


def test_preflight_ok_on_gpu(fakes):
    assert whisper.preflight(wcfg()) == []


def test_preflight_ffmpeg_missing(fakes, monkeypatch):
    monkeypatch.setattr(checks.shutil, "which", lambda name: None)
    with pytest.raises(ScribeError, match="ffmpeg: not found in PATH"):
        whisper.preflight(wcfg())


def test_preflight_whisperx_missing(fakes, monkeypatch):
    monkeypatch.setitem(sys.modules, "whisperx", None)
    with pytest.raises(InstallError, match="WhisperX: not installed"):
        whisper.preflight(wcfg())


def test_preflight_cpu_build_of_torch(fakes):
    fakes.torch.version.cuda = None
    with pytest.raises(InstallError, match="PyTorch 2.8.0: a CPU-only build"):
        whisper.preflight(wcfg())


def test_preflight_no_cuda(fakes):
    fakes.cuda = False
    with pytest.raises(InstallError, match="GPU: not seen"):
        whisper.preflight(wcfg())


def test_preflight_cpu_device_warns(fakes):
    fakes.cuda = False
    fakes.torch.version.cuda = None
    [warning] = whisper.preflight(wcfg(device="cpu"))
    assert "very slow" in warning


def test_preflight_cpu_with_float16_is_an_error(fakes):
    with pytest.raises(ScribeError, match="float16.*int8") as e:
        whisper.preflight(wcfg(device="cpu", compute_type="float16"))
    assert not isinstance(e.value, InstallError)  # a settings error


@pytest.mark.parametrize("found", ["3.4.2", "4.0.0"])
def test_preflight_warns_on_untested_whisperx(fakes, found):
    fakes.whisperx_version = found
    [warning] = whisper.preflight(wcfg())
    assert found in warning and "tested" in warning


def test_preflight_newer_3x_is_fine(fakes):
    fakes.whisperx_version = "3.9.1"
    assert whisper.preflight(wcfg()) == []


def test_preflight_unknown_whisperx_version_is_fine(fakes):
    fakes.whisperx_version = "unknown"
    assert whisper.preflight(wcfg()) == []


# --- versions ---------------------------------------------------------------


def test_versions(monkeypatch):
    versions = {"whisperx": "3.8.6", "torch": "2.8.0+cu128"}
    monkeypatch.setattr(checks, "installed_version", lambda pkg: versions[pkg])
    monkeypatch.setattr(ffmpeg, "version", lambda: "7.1.1")
    assert whisper.versions() == {
        "whisperx": "3.8.6",
        "torch": "2.8.0+cu128",
        "ffmpeg": "7.1.1",
    }


# --- Transcriber ------------------------------------------------------------


def test_transcriber_sets_prompt_through_public_options(fakes):
    t = whisper.Transcriber(
        wcfg(prompt_max_tokens=3), ["Vorgrad", "Ostromir", "Czarna Wieża"]
    )
    assert t.initial_prompt == "Vorgrad, Ostromir"
    assert t.prompt_cut == ["Czarna Wieża"]
    first, second = fakes.loads
    assert first["asr_options"] is None
    assert second["asr_options"] == {"initial_prompt": "Vorgrad, Ostromir"}
    assert second["model"] is not None  # reuses the loaded weights
    assert t.warnings == []


def test_transcriber_passes_settings_through(fakes):
    whisper.Transcriber(
        wcfg(
            model="medium",
            device="cpu",
            compute_type="int8",
            language="pl",
            align_model="org/model",
        ),
        [],
    ).transcribe([0.0] * 16000, "gm_account")
    load = fakes.loads[0]
    assert (load["arch"], load["device"], load["compute_type"], load["language"]) == (
        "medium",
        "cpu",
        "int8",
        "pl",
    )
    assert fakes.align_loads == [("pl", "cpu", "org/model")]


def test_transcriber_estimates_tokens_without_tokenizer(fakes):
    fakes.tokenizer = False
    t = whisper.Transcriber(wcfg(prompt_max_tokens=4), ["Vorgrad", "Ostromir"])
    # estimate: 1 token per 2 characters -> "Vorgrad" is 4, both are 9
    assert t.initial_prompt == "Vorgrad"
    assert t.prompt_cut == ["Ostromir"]
    assert any("estimated" in w for w in t.warnings)


def test_transcribe_returns_segments_and_reports_progress(fakes):
    t = whisper.Transcriber(wcfg(), [])
    seen = []
    [s] = t.transcribe([0.0] * 16000, "gm_account", on_progress=seen.append)
    assert s.text == "To jest test" and s.avg_logprob == -0.2
    # transcription 50 % and 100 % -> 45, 90; alignment 100 % -> 100; done -> 100
    assert seen == [45.0, 90.0, 100.0, 100.0]
    assert all(a <= b for a, b in zip(seen, seen[1:], strict=False))


def test_transcribe_without_progress_support(fakes):
    fakes.progress = False
    t = whisper.Transcriber(wcfg(), [])
    seen = []
    [s] = t.transcribe([0.0] * 16000, "gm_account", on_progress=seen.append)
    assert s.text == "To jest test"
    assert seen == [100.0, 100.0]  # transcription silent; alignment 100, done 100


def test_settings_record_prompt(fakes):
    t = whisper.Transcriber(wcfg(), ["Vorgrad"])
    settings = t.settings()
    assert settings["initial_prompt"] == "Vorgrad"
    assert settings["prompt_cut"] == []
    assert settings["model"] == "large-v3"


def test_out_of_memory(fakes):
    fakes.load_error = RuntimeError("CUDA failed with error out of memory")
    with pytest.raises(ScribeError, match="smaller transcribe.whisper.batch_size"):
        whisper.Transcriber(wcfg(), [])


def test_no_alignment_model_for_language(fakes):
    fakes.align_error = ValueError("No default align-model for language: xx")
    t = whisper.Transcriber(wcfg(language="xx"), [])
    with pytest.raises(ScribeError, match="transcribe.whisper.align_model"):
        t.transcribe([0.0] * 16000, "gm_account")


def test_other_errors_keep_their_type(fakes):
    fakes.load_error = RuntimeError("something else")
    with pytest.raises(RuntimeError, match="something else"):
        whisper.Transcriber(wcfg(), [])


def test_whisperx_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "whisperx", None)
    with pytest.raises(InstallError, match="WhisperX: not installed"):
        whisper.Transcriber(wcfg(), [])


def test_models_load_offline(fakes):
    whisper.Transcriber(wcfg(), ["Vorgrad"]).transcribe([0.0] * 16000, "gm_account")
    assert all(load["local_files_only"] is True for load in fakes.loads)
    assert fakes.align_cache_only is True


def test_damaged_model_file_is_a_clear_error(fakes):
    fakes.align_error = RuntimeError(
        "PytorchStreamReader failed reading zip archive: "
        "failed finding central directory"
    )
    t = whisper.Transcriber(wcfg(), [])
    with pytest.raises(ScribeError, match="damaged"):
        t.transcribe([0.0] * 16000, "gm_account")


def test_preflight_quiets_library_logs(fakes):
    import logging

    whisper.preflight(wcfg())
    # The checkpoint-upgrade notice comes from this exact logger; Lightning sets
    # its own "lightning.pytorch" logger to INFO on import, so the parent
    # loggers alone are not enough.
    # WhisperX's per-segment alignment warnings are hidden too (ERROR); the
    # affected words show up as estimated in the report.
    assert logging.getLogger("whisperx").level == logging.ERROR
    names = (
        "pyannote",
        "lightning.pytorch.utilities.migration.utils",
        "pytorch_lightning.utilities.migration.utils",
    )
    for name in names:
        assert logging.getLogger(name).level == logging.WARNING


# --- one model on the GPU at a time ----------------------------------------------


def test_only_one_model_on_the_gpu_at_a_time(fakes):
    t = whisper.Transcriber(wcfg(), ["Vorgrad"])
    loads_at_start = len(fakes.loads)  # base + with prompt
    t.transcribe([0.0] * 16000, "gm_account")
    t.transcribe([0.0] * 16000, "player_one")
    # Whisper is unloaded after each track and loaded again for the next one,
    # always with the glossary prompt; the alignment model once per track.
    later = fakes.loads[loads_at_start:]
    assert len(later) == 1
    assert later[0]["asr_options"] == {"initial_prompt": "Vorgrad"}
    assert len(fakes.align_loads) == 2
    assert fakes.freed >= 4  # GPU cache emptied after every unload


def test_phases_reported_per_track(fakes):
    t = whisper.Transcriber(wcfg(), [])
    first, second = [], []
    t.transcribe([0.0] * 16000, "gm_account", on_phase=first.append)
    t.transcribe([0.0] * 16000, "player_one", on_phase=second.append)
    assert first == ["voice detection", "transcribing", "loading alignment model",
                     "aligning"]  # fmt: skip
    assert second[0] == "loading Whisper" and second[1:] == first


def test_low_gpu_memory_warning(fakes):
    fakes.free_bytes = 1024**3  # 1 GB left after loading Whisper
    t = whisper.Transcriber(wcfg(), [])
    assert any("Sysmem Fallback" in w and "batch_size" in w for w in t.warnings)


def test_enough_gpu_memory_no_warning(fakes):
    assert whisper.Transcriber(wcfg(), []).warnings == []


def test_no_memory_check_on_cpu(fakes):
    fakes.free_bytes = 0
    assert whisper.Transcriber(wcfg(device="cpu"), []).warnings == []


# --- hidden library warnings -------------------------------------------------------

SYMLINKS = (
    "`huggingface_hub` cache-system uses symlinks by default to efficiently store "
    "duplicated files but your machine does not support them in /cache."
)
XET = (
    "Xet Storage is enabled for this repo, but the 'hf_xet' package is not "
    "installed. Falling back to regular HTTP download."
)


def test_hidden_warnings_catch_the_known_ones():
    hf_log = logging.getLogger("huggingface_hub.file_download")
    with whisper.hidden_warnings() as hidden:
        warnings.warn(SYMLINKS, stacklevel=1)
        warnings.warn("\ntorchcodec is not installed correctly\nso", stacklevel=1)
        hf_log.warning(XET)
    assert hidden == [
        "library warning: " + SYMLINKS,
        "library warning: torchcodec is not installed correctly so",
        "library warning: " + XET,
    ]


def test_hidden_warnings_hide_the_home_folder():
    home = str(Path.home())
    text = f"torchcodec is not installed correctly: {home}{os.sep}env{os.sep}x.dll"
    with whisper.hidden_warnings() as hidden:
        warnings.warn(text, stacklevel=1)
    assert hidden == [
        f"library warning: torchcodec is not installed correctly: ~{os.sep}env"
        f"{os.sep}x.dll"
    ]


def test_hidden_warnings_let_other_warnings_through():
    with pytest.warns(UserWarning, match="something new"):
        with whisper.hidden_warnings() as hidden:
            warnings.warn("something new", stacklevel=1)
    assert hidden == []


def test_hidden_warnings_restore_the_originals():
    hf_log = logging.getLogger("huggingface_hub.file_download")
    show, filters = warnings.showwarning, list(hf_log.filters)
    with pytest.raises(RuntimeError), whisper.hidden_warnings():
        raise RuntimeError("boom")
    assert warnings.showwarning is show and hf_log.filters == filters
