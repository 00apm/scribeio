# scribe

Turn a [Craig](https://craig.chat) recording of your Discord tabletop RPG
session into evenly balanced session audio and a readable transcript with
everyone's name, plus a quality report. Everything runs on your own
computer.

- Craig records one track per person, so every line of the transcript has
  the right speaker, even when people talk over each other.
- Whisper's hallucinations (lines nobody said) are found and removed; the
  report lists every removal.
- A misheard name is fixed with one line in a glossary and a rebuild that
  takes seconds, without the GPU.

Transcription runs on [WhisperX](https://github.com/m-bain/whisperX) and
the Whisper `large-v3` model.

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
