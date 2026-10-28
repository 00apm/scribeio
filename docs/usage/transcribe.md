# Transcribing a recording

Turn a recording into a readable transcript with everyone's name. Making
it better (names of your world, language, model, line length) is on
[tuning the transcription](transcribe-tuning.md).

## Configure the transcription

```
scribe init "craig-session-7.zip"
```

The transcription needs `scribe.yaml` with each person's name next to
their Discord account, and the language if it is not English;
`glossary.txt` with the names of your world is optional. How to fill them
in: [create the settings file](../settings.md#create-the-settings-file).
Without `scribe.yaml`, `scribe transcribe` offers to create it. The input
is a Craig download or a folder of FLAC tracks
([input recordings](../recordings.md)).

## Transcribe

```
scribe transcribe "craig-session-7.zip"
```

Each player's track is transcribed on its own, so every line has the right
speaker even when people talk over each other. The music track is left out.
A 10-minute recording took about a minute and a half on an 8 GB NVIDIA
GPU. When all goes well, the run looks like this, and its last line names
the transcript and the report:

```
using config scribe.yaml (next to the input)
✔ check ffmpeg, WhisperX, CUDA (26.8s)
✔ check models (0.0s)
✔ open craig-session-7.zip (0.0s)
music (not transcribed): Kenku
✔ load Whisper large-v3 (pl) on cuda (2.8s)
✔ gm_account: 57 segments (15.8s)
✔ player_one: 56 segments (20.1s)
…
✔ write 2026-09-22_transcript.md (0.0s)
2026-09-22_transcript.md: 87 lines, 5 segments dropped, 0 possible bleed - see 2026-09-22_transcript-report.md
```

If the outputs already exist, scribeio stops; `--force` overwrites them.
`--name` and `-o` change the name and the folder, as for the audio.

### Download the models on the first run

The first transcription needs two speech models: the Whisper model
(`large-v3` by default, 2.9 GB;
[choose the model](transcribe-tuning.md#choose-the-model-and-the-device))
and a smaller word-timing model for the
language (up to about 1.2 GB). scribeio asks before downloading each one,
with its size:

```
⚠ Whisper model large-v3 (2.9 GB) is not downloaded
  [d] download it
  [a] abort
Choose (d, a) [a]:
```

`--yes` downloads without asking. After that, transcription works offline.
`scribe check` shows which models are on disk.

## Read the results

| File | What it is |
|---|---|
| `2026-09-22_transcript.md` | the transcript |
| `2026-09-22_transcript-report.md` | the quality report |
| `2026-09-22_transcript.json` | everything Whisper found, for rebuilding the transcript |

The transcript starts with the session's date, start time, length and
speakers, then one line per utterance, in time order:

```
---
date: 2026-09-22
start: "20:00"
duration: "3:41:12"
speakers: ["Game Master", "Player One", "Player Two"]
source: "craig-session-7.zip"
---

[00:00:00] Game Master: The mists part, and the road to Barovia lies ahead.
[00:00:04] Player One: We keep walking. Slowly.
[00:00:06] Player Two: Agreed.
[00:00:07] Game Master: Far above, a castle stands against the grey sky.
```

A new line starts when a speaker pauses for 1.5 seconds
([choose when a new line starts](transcribe-tuning.md#choose-when-a-new-line-starts));
a short reply from someone else lands in the middle of a long speech,
where it was said.

### Hallucinations are removed

Whisper hallucinates: it sometimes writes things nobody said. Typical
cases are a "thanks for watching" in a silent stretch, the same phrase
repeated over and over, and lines made only of hesitations ("um", "yyy").
scribeio looks for these and deletes them from the transcript. The report
lists every deleted line with the reason, so you can check that nothing
real went missing. To remove more or fewer of them, see
[hallucinations and hesitations](transcribe-tuning.md#advanced-hallucinations-and-hesitations).

The report also shows, per speaker, how many lines were kept and dropped,
and lists "possible bleed": the same speech found on two tracks, usually
one mic picking up another person's speakers.

## Make the transcript better

A name comes out wrong, the language is not English, the run is too slow,
lines are too long or too short: see
[tuning the transcription](transcribe-tuning.md). A misheard name is fixed
with one line in `glossary.txt` and a rebuild that takes seconds
([fix a misspelt name](transcribe-tuning.md#fix-a-misspelt-name)).

## Transcribe the processed tracks

`scribe master --tracks` writes each track after audio mastering
([write each processed track](audio.md#write-each-processed-track)).
That folder can be transcribed like the recording:

```
scribe transcribe "2026-09-22_tracks"
```

## Troubleshooting

### CUDA out of memory

```
CUDA out of memory; try a smaller transcribe.whisper.batch_size (now 4)
```

The GPU ran out of memory. Lower `batch_size` in `scribe.yaml`:

```yaml
transcribe:
  whisper:
    batch_size: 2
```

If that is not enough, see [GPU](../troubleshooting.md#gpu) for the settings that
need less memory.

### Transcription is much slower than expected (Windows)

When GPU memory is full, the NVIDIA driver quietly moves the rest into
normal RAM ("Shared GPU memory" in Task Manager rises above zero), and the
work slows down several times, without any error. scribeio warns when little
GPU memory is left after loading Whisper (`only 1.4 GB GPU memory free
after loading Whisper; …`). Lower `batch_size` as above, or set *NVIDIA
Control Panel → Manage 3D settings → CUDA - Sysmem Fallback Policy* to
*Prefer No Sysmem Fallback*: you then get the clear out-of-memory error
instead.
