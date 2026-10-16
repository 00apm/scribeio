"""Raw session data: what `transcribe` measures and `merge` reads back.

Everything here is a measurement, never a decision: filtering, speaker names
and line building happen later in the pipeline, so `scribe merge` can redo
them from the saved JSON without audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Word:
    """One word with its timing.

    Attributes:
        word: The word as Whisper wrote it, punctuation included.
        start: Start in seconds from the session start.
        end: End in seconds from the session start.
        score: Alignment confidence, if known.
        voiced: True if the speaker's own track is loud enough under the word.
        estimated: True if alignment gave no timestamp and it was interpolated.
    """

    word: str
    start: float
    end: float
    score: float | None = None
    voiced: bool = True
    estimated: bool = False


@dataclass
class Segment:
    """One Whisper segment of one track.

    Attributes:
        account: Track account (as in the file name).
        start: Start in seconds from the session start.
        end: End in seconds from the session start.
        text: Segment text.
        words: The segment's words, in order.
        avg_logprob: Whisper's average log probability, if known.
        level_db: Mean dBFS of the voiced frames, or None if none was voiced.
    """

    account: str
    start: float
    end: float
    text: str
    words: list[Word]
    avg_logprob: float | None = None
    level_db: float | None = None


@dataclass
class TrackInfo:
    """One track of the recording, as stored in the JSON.

    Attributes:
        account: Track account (as in the file name).
        kind: ``"player"`` or ``"music"``.
        file: File name only, never a full path.
        transcribed: Whether the track was transcribed.
    """

    account: str
    kind: str
    file: str
    transcribed: bool


@dataclass
class RawSession:
    """Everything `transcribe` stores in ``<name>_transcript.json``.

    Attributes:
        date: Session date, ``YYYY-MM-DD``, local time.
        start: Session start, ISO 8601 local time with UTC offset.
        duration: Length of the longest track in seconds.
        source: Zip or folder name only.
        whisper: Whisper settings used, plus ``initial_prompt`` and
            ``prompt_cut`` (glossary names left out of the prompt).
        tracks: All tracks of the recording.
        segments: Segments of all transcribed tracks.
        scribe: scribe version that wrote the file.
        versions: Versions of whisperx, torch and ffmpeg used.
        warnings: Warnings about the transcription run itself (device,
            GPU memory, untested WhisperX), for the report.
    """

    date: str
    start: str
    duration: float
    source: str
    whisper: dict[str, Any]
    tracks: list[TrackInfo]
    segments: list[Segment]
    scribe: str = ""
    versions: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
