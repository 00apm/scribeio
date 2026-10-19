<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/00apm/scribeio/main/docs/images/scribeio-logo-dark.svg">
    <img alt="scribeio logo" src="https://raw.githubusercontent.com/00apm/scribeio/main/docs/images/scribeio-logo-light.svg" width="360">
  </picture>
</p>

# scribeio

Turn [Craig](https://craig.chat) (Discord) recordings of your TTRPG
sessions into balanced audio and a transcript ready for your session
notes.

Craig records one track per person. scribeio works on those tracks:

- **Audio mastering:** each voice is evened out and brought to the same
  level as the others, the music is kept quieter than the voices so it
  never drowns them out, and everything is mixed into one file.
- **Transcript:** each track is transcribed by Whisper, a speech-to-text
  model that runs on your own computer
  (through [WhisperX](https://github.com/m-bain/whisperX)), and the lines
  are put together in time order with each person's name.

What you get from working per track, locally:

- Every line has the right speaker, even when people talk over each
  other.
- Lines Whisper made up (hallucinations, e.g. "thanks for watching" in a
  silence) are found and removed; a report lists each one.
- The names of your world are taught to Whisper in a glossary; a misheard
  name is fixed with one line and a rebuild that takes seconds.
- Your recordings never leave your computer: your players' voices and
  words stay private, and no online service can keep them or use them to
  train its AI models.

**Not a TTRPG session, or not Craig?** A podcast or a meeting recorded on
Discord with Craig works the same way, and so does any other recording
with one track per person
([input recordings](https://github.com/00apm/scribeio/blob/main/docs/recordings.md#not-using-craig)).

## Requirements

- Windows, macOS or Linux.
- [ffmpeg](https://ffmpeg.org) 6.1 or newer.
- For transcription: an NVIDIA GPU (tested on 8 GB). Without one it works,
  very slowly. Audio mastering needs no GPU.
- Disk: up to about 12 GB (the install and the speech models).

## Install

- [Quick install](https://github.com/00apm/scribeio/blob/main/docs/install.md):
  a few commands, no Python knowledge needed.
- [Install into your own Python environment](https://github.com/00apm/scribeio/blob/main/docs/install-own-env.md):
  pip, Python 3.10–3.13.

## Quick start

In the folder with the Craig download:

```
scribe init "craig-session-7.zip"
```

Write each person's name next to their Discord account in `scribe.yaml`,
then make the session audio and the transcript:

```
scribe master "craig-session-7.zip"
scribe transcribe "craig-session-7.zip"
```

`scribe check` shows what is installed and what is missing.

The files land next to the recording, named by the session date:

| File | What it is |
|---|---|
| `2026-09-22_master.mp3` | the session audio |
| `2026-09-22_transcript.md` | the transcript, one line per utterance |
| `2026-09-22_transcript-report.md` | the quality report |
| `2026-09-22_transcript.json` | everything Whisper found, for rebuilding the transcript |

The quick start runs everything on its defaults. The guides below show
how to make the result your own.

## Usage guide

Read in this order the first time; later, go straight to the part you
need.

1. **[Input recordings](https://github.com/00apm/scribeio/blob/main/docs/recordings.md):**
   which Craig downloads work, where scribeio puts its files, and how to
   use tracks that did not come from Craig.
2. **[Audio mastering](https://github.com/00apm/scribeio/blob/main/docs/usage/audio.md):**
   listen to a short preview after each step, change how the voices and
   the music sound, adjust one player only, and choose the file format.
3. **[Transcribing a recording](https://github.com/00apm/scribeio/blob/main/docs/usage/transcribe.md):**
   set names and the language, download the models, read the transcript
   and the report, and teach Whisper the names of your world with a
   glossary.

All documentation pages, install included:
[docs](https://github.com/00apm/scribeio/blob/main/docs/README.md).

## Development

```
pip install -e ".[transcribe,dev]"
pytest -m "not gpu"
mypy
ruff format src tests && ruff check src tests
```

Tests use synthetic data only; no real recordings are in the repository.

## License

scribeio is free and open source under the
[MIT License](https://github.com/00apm/scribeio/blob/main/LICENSE): you
may use, change and share it, also commercially, as long as the licence
text stays with it. ffmpeg, WhisperX and the speech models are installed
separately and come with their own licences.
