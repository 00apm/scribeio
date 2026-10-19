"""Real FLAC through WhisperX's decoder and the energy measurement (spec 12.2).

Needs ffmpeg in PATH and WhisperX importable; skipped otherwise.
"""

import subprocess

import pytest

from helpers import seg

pytestmark = pytest.mark.ffmpeg


def test_real_flac_through_load_audio_and_energy(tmp_path):
    pytest.importorskip("whisperx")
    from scribeio.transcribe import whisper
    from scribeio.transcribe.energy import annotate

    # 1 s of 440 Hz tone, then 1 s of silence; stereo 48 kHz like Craig.
    flac = tmp_path / "1-gm_account.flac"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=48000",
            "-af", "apad=pad_dur=1", "-ac", "2", "-c:a", "flac", str(flac),
        ],
        check=True,
    )  # fmt: skip
    audio = whisper.load_audio(flac)
    assert audio.dtype.name == "float32"
    assert len(audio) == pytest.approx(32000, abs=320)  # 2 s at 16 kHz
    loud = seg("gm_account", 0.2, "raz dwa")
    quiet = seg("gm_account", 1.2, "trzy cztery")
    annotate([loud, quiet], audio, energy_db=-50, frame_ms=50)
    assert all(w.voiced for w in loud.words)
    assert not any(w.voiced for w in quiet.words)
