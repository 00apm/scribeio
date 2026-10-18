# Transcribing a session

Turn a session recording into a readable transcript with everyone's name,
and fix the names Whisper gets wrong.

## Prepare the names

```
scribe init "craig-session-7.zip"
```

The input is the Craig zip or a folder of FLAC tracks
([what scribe reads](../recordings.md)). This writes two files next to it:

- `scribe.yaml`, with the recording's Discord accounts listed under
  `speakers:`. Write each person's name after their account; the
  transcript uses these names. Set the language here too (`language: pl`).
- `glossary.txt`, for the names of your world (see
  [Teach Whisper your names](#teach-whisper-your-names)).

Every scribe command uses `scribe.yaml` by itself when it sits next to the
input.

## Transcribe

```
scribe transcribe "craig-session-7.zip"
```

Each player's track is transcribed on its own, so every line has the right
speaker even when people talk over each other. The music bot is left out.
A 10-minute recording took about a minute and a half on an 8 GB NVIDIA
GPU. A run:

```
using config scribe.yaml (next to the input)
✔ check ffmpeg, WhisperX, CUDA (26.8s)
✔ check models (0.0s)
✔ open craig-session-7.zip (0.0s)
music (not transcribed): Kenku
✔ load Whisper large-v3 (pl) on cuda (2.8s)
✔ gm.account: 57 segments (15.8s)
✔ player.one: 56 segments (20.1s)
…
✔ write 2026-09-22_transcript.md (0.0s)
2026-09-22_transcript.md: 87 lines, 5 segments dropped, 0 possible bleed - see 2026-09-22_transcript-report.md
```

If the outputs already exist, scribe stops; `--force` overwrites them.
`--name` and `-o` change the name and the folder, as for the audio.

### Download the models on the first run

The first transcription needs two speech models: Whisper (`large-v3`,
2.9 GB) and a smaller word-timing model for the language (up to about
1.2 GB). scribe asks before downloading each one, with its size:

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

[00:00:00] Game Master: The gate of Vorgrad is closed, and the guards look tired.
[00:00:04] Player One: We wait until they change shifts.
[00:00:06] Player Two: Agreed.
[00:00:07] Game Master: The bell rings, and two new guards walk up the road.
```

A new line starts when a speaker pauses for 1.5 seconds; a short reply
from someone else lands in the middle of a long speech, where it was said.

Whisper sometimes writes things nobody said: a "thanks for watching" in a
silent stretch, a phrase repeated over and over, hesitations ("um", "yyy").
scribe drops these, and the report lists every dropped line with the
reason, so you can check that nothing real went missing. It also shows,
per speaker, how many lines were kept and dropped, and lists "possible
bleed": the same speech found on two tracks, usually one mic picking up
another person's speakers.

## Teach Whisper your names

Whisper knows common words, not the names of your world: "Vorgrad" comes
out as "Worgrad", or differently each time. `glossary.txt` fixes this in
two ways:

```
# names Whisper should listen for, most important first
Vorgrad
Ostromir

# corrections: wrong -> right
Ostromirz -> Ostromir
Worgrad* -> Vorgrad
```

- **A name on its own line** is given to Whisper before it listens, which
  helps it hear the name right. The space is limited
  (`prompt_max_tokens`, 200 by default), so put the most important names
  first; scribe warns about names left out.
- **`wrong -> right`** replaces the wrong spelling in the transcript, as a
  whole word or phrase, ignoring case.
- **`stem* -> right`** replaces any word that starts with the stem and
  keeps its ending: `Worgradzie` becomes `Vorgradzie`. Useful in languages
  where names change their endings.

The right side of a correction counts as a name for Whisper too.

### Fix a misspelt name

When the transcript has a name wrong:

1. Add a correction to `glossary.txt`, e.g. `Ostromirz -> Ostromir`.
2. Rebuild the transcript from the `.json`:

   ```
   scribe merge "2026-09-22_transcript.json"
   ```

This takes seconds and needs no GPU. `scribe merge` rebuilds the
transcript and the report, never the `.json`; it also picks up changed
names under `speakers:`. New prompt names help only the next
`scribe transcribe`.

## Choose the language

```yaml
transcribe:
  whisper:
    language: pl
```

Any language code Whisper knows works (`en`, `de`, `fr`, …). English and
Polish are tuned and tested: for those, scribe knows Whisper's typical
made-up phrases and the hesitation sounds. For another language, add the
phrases you find in the report's dropped lines, or in the transcript, to
`transcribe.cleanup.phrases` in `scribe.yaml` (a list you write replaces
the default one).

## Transcribe the processed tracks

`scribe master --tracks` writes each track after audio mastering
([write each processed track](audio.md#write-each-processed-track)).
That folder can be transcribed like the recording:

```
scribe transcribe "2026-09-22_tracks"
```

## Troubleshooting

### CUDA out of memory

The GPU ran out of memory. Lower `batch_size` in `scribe.yaml`:

```yaml
transcribe:
  whisper:
    batch_size: 2
```

If that is not enough, see [GPU](../install.md#gpu) for the settings that
need less memory.
