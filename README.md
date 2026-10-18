# scribe

Turn a [Craig](https://craig.chat) recording of your Discord tabletop RPG
session into evenly balanced session audio and a readable transcript with
everyone's name.

Craig records one track per person. scribe works on those tracks:

- **Audio:** each voice is evened out and brought to the same level as
  the others, the music bot is set under the voices, and everything is
  mixed into one file.
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
- Your recordings never leave your computer.

**Not a TTRPG session, or not Craig?** A podcast or a meeting recorded on
Discord with Craig works the same way, and so does any other recording
with one track per person
([what scribe reads](docs/recordings.md#not-using-craig)).

## Requirements

- Windows, macOS or Linux.
- [ffmpeg](https://ffmpeg.org) 6.1 or newer.
- For transcription: an NVIDIA GPU (tested on 8 GB). Without one it works,
  very slowly. Audio mastering needs no GPU.
- Disk: up to about 12 GB (the install and the speech models).

## Install

- [Quick install](docs/install.md): a few commands, no Python knowledge
  needed.
- [Install into your own Python environment](docs/install-own-env.md):
  pip, Python 3.10–3.13.

## Quick start

In the folder with the Craig download:

```
scribe init "craig-session-7.zip"
```

Write each person's name next to their Discord account in `scribe.yaml`,
then:

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

## Documentation

[All pages](docs/README.md): installing, which recordings work, audio
mastering, transcription.

## Development

```
pip install -e ".[transcribe,dev]"
pytest -m "not gpu"
mypy
ruff format src tests && ruff check src tests
```

Tests use synthetic data only; no real recordings are in the repository.

## License

MIT
