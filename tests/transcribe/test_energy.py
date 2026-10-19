"""Tests for per-word voiced flags and per-segment levels."""

import tracemalloc

import numpy as np
import pytest

from scribeio.transcribe.energy import annotate, frame_levels

from helpers import seg, silence, tone

TONE_DB = 20 * np.log10(0.1 / np.sqrt(2))  # RMS of a 0.1 sine: about -23 dBFS


def test_frame_levels_tone_and_silence():
    levels = frame_levels(np.concatenate([tone(1.0, 0.1), silence(1.0)]), 50)
    assert len(levels) == 40
    assert levels[:20] == pytest.approx(TONE_DB, abs=0.5)
    assert (levels[20:] == -120).all()


def test_frame_levels_partial_last_frame():
    assert len(frame_levels(tone(1.01), 50)) == 21


def test_frame_levels_empty_audio():
    assert len(frame_levels(silence(0), 50)) == 0


def _padded_levels(audio, frame_ms, sample_rate=16_000):
    """The original whole-track computation (float64, padded), for comparison."""
    n = int(sample_rate * frame_ms / 1000)
    x = np.asarray(audio, dtype=np.float64)
    frames = np.pad(x, (0, (-len(x)) % n)).reshape(-1, n)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return np.maximum(20 * np.log10(np.maximum(rms, 1e-12)), -120.0)


@pytest.mark.parametrize("seconds", [0.03, 1.0, 37.013])
def test_frame_levels_equal_whole_track_computation(seconds):
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(int(16_000 * seconds)) * 0.05).astype(np.float32)
    audio[: len(audio) // 3] = 0  # digital silence too
    assert np.array_equal(frame_levels(audio, 50), _padded_levels(audio, 50))


def test_frame_levels_memory_stays_small():
    audio = np.full(16_000 * 600, 0.01, dtype=np.float32)  # 10 min, 38 MB
    tracemalloc.start()
    try:
        frame_levels(audio, 50)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < audio.nbytes / 4


def test_words_in_tone_voiced_words_in_silence_not():
    audio = np.concatenate([tone(1.0), silence(1.0)])
    # words: 0.2-0.6, 0.6-1.0, 1.0-1.4, 1.4-1.8
    s = seg("gm_account", 0.2, "raz dwa trzy cztery", word_len=0.4)
    annotate([s], audio, energy_db=-50, frame_ms=50)
    assert [w.voiced for w in s.words] == [True, True, False, False]


def test_word_partly_over_tone_is_voiced():
    audio = np.concatenate([tone(1.0), silence(1.0)])
    s = seg("gm_account", 0.95, "słowo", word_len=0.5)  # 0.95-1.45
    annotate([s], audio, energy_db=-50, frame_ms=50)
    assert s.words[0].voiced


def test_threshold_is_strictly_greater():
    audio = np.full(16000, 0.01, dtype=np.float32)  # -40 dBFS
    s1, s2 = seg("gm_account", 0.1, "a"), seg("gm_account", 0.1, "a")
    annotate([s1], audio, energy_db=-39.9, frame_ms=50)
    annotate([s2], audio, energy_db=-40.1, frame_ms=50)
    assert not s1.words[0].voiced
    assert s2.words[0].voiced


def test_level_db_is_mean_of_voiced_frames():
    audio = np.concatenate([tone(1.0, 0.1), silence(1.0)])
    s = seg("gm_account", 0.0, "a b c d e", word_len=0.4)  # 0-2 s, half tone
    annotate([s], audio, energy_db=-50, frame_ms=50)
    assert s.level_db == pytest.approx(TONE_DB, abs=0.5)


def test_level_db_none_when_silent():
    s = seg("gm_account", 0.0, "a b", word_len=0.4)
    annotate([s], silence(2.0), energy_db=-50, frame_ms=50)
    assert s.level_db is None
    assert not any(w.voiced for w in s.words)


def test_word_past_end_of_audio_not_voiced():
    s = seg("gm_account", 5.0, "a")
    annotate([s], tone(1.0), energy_db=-50, frame_ms=50)
    assert not s.words[0].voiced
    assert s.level_db is None


def test_zero_length_word_uses_one_frame():
    s = seg("gm_account", 0.5, "a", word_len=0.0)
    annotate([s], tone(1.0), energy_db=-50, frame_ms=50)
    assert s.words[0].voiced
