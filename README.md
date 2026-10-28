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
- For transcription: an NVIDIA GPU, GTX 16xx / RTX 20xx or newer, 6 GB or
  more (8 GB recommended). Without one it runs on the CPU, very slowly
  (`transcribe.whisper.device: cpu` in `scribe.yaml`). Audio mastering needs no GPU.
- Disk: up to about 12 GB with transcription (the install and the speech
  models); audio mastering alone needs well under 1 GB.

## Install

- [Quick install](https://github.com/00apm/scribeio/blob/main/docs/install.md):
  a few commands, no Python knowledge needed.
- [Install into your own Python environment](https://github.com/00apm/scribeio/blob/main/docs/install-own-env.md):
  pip, Python 3.10–3.13.
- [Uninstall](https://github.com/00apm/scribeio/blob/main/docs/uninstall.md):
  remove scribeio and what it downloaded (the speech models, PyTorch,
  ffmpeg).

## Quick start

In the folder with the Craig download:

```
scribe init "craig-session-7.zip"
```

This creates scribeio's settings for this recording: `scribe.yaml`, with
the Discord accounts from the recording already listed, and `glossary.txt`
for the names of your world. Write each person's name next to their
account in `scribe.yaml`, then make the session audio and the transcript:

```
scribe master "craig-session-7.zip"
scribe transcribe "craig-session-7.zip"
```

`scribe check` shows what is installed and what is missing.

The files land next to the recording, named by the session date:

| File | What it is |
|---|---|
| `2026-09-22_master.mp3` | the session audio, every voice at the same level |
| `2026-09-22_transcript.md` | the transcript, one line per utterance |
| `2026-09-22_transcript.json` | everything Whisper found, for rebuilding the transcript without the GPU |
| `2026-09-22_transcript-report.md` | the quality report: what was removed and why |

A piece of the transcript:

```
[01:12:05] Alice (GM): The stairs end at a heavy door. Cold air seeps through the cracks, and from somewhere below comes a slow
[01:12:11] Sam: Please don't say scratching.
[01:12:12] Alice (GM): scratching. Something is scratching at the wood.
[01:12:16] Robin: I put my ear to the door.
[01:12:17] Sam: We are not opening that door.
[01:12:19] Alice (GM): The scratching stops. Then, very quietly, something knocks. Three times.
[01:12:25] Sam: Nope. I'm going back up.
[01:12:25] Robin: I open the door.
```

The quick start runs everything on its defaults. The guides below show
how to make the result your own.

## Usage guide

Each guide covers one part; go to the one you need.

- **[Input recordings](https://github.com/00apm/scribeio/blob/main/docs/recordings.md):**
  which Craig downloads work, where scribeio puts its files, and how to
  use tracks that did not come from Craig.
- **[Your settings file](https://github.com/00apm/scribeio/blob/main/docs/settings.md):**
  why `scribe.yaml` exists, how to read it and change a setting without
  breaking it.
- **[Audio mastering](https://github.com/00apm/scribeio/blob/main/docs/usage/audio.md):**
  make the session audio and listen to a short preview after each step;
  [tuning the audio](https://github.com/00apm/scribeio/blob/main/docs/usage/audio-tuning.md):
  voice and music levels, one player only, the file format.
- **[Transcribing a recording](https://github.com/00apm/scribeio/blob/main/docs/usage/transcribe.md):**
  download the models, read the transcript and the report;
  [tuning the transcription](https://github.com/00apm/scribeio/blob/main/docs/usage/transcribe-tuning.md):
  the names of your world and corrections in a glossary, the language,
  the model; [rebuild the transcript](https://github.com/00apm/scribeio/blob/main/docs/usage/merge.md)
  in seconds after a fix.

Every page, with a "where do I find…" guide:
[docs](https://github.com/00apm/scribeio/blob/main/docs/README.md).

## Development

```
git clone https://github.com/00apm/scribeio.git
cd scribeio
pip install -e ".[dev]"
pytest -m "not gpu"
mypy
ruff format src tests && ruff check src tests
```

The tests need ffmpeg 6.1 or newer with the mp3 encoder on PATH, but not
WhisperX, PyTorch or a GPU: transcription is tested with stand-ins. Tests
use synthetic data only; no real recordings are in the repository (the
`gpu` tests need a local recording; `-m "not gpu"` leaves them out).

To run real transcriptions from the checkout, install PyTorch with CUDA
first
([Install into your own Python environment](https://github.com/00apm/scribeio/blob/main/docs/install-own-env.md#install)),
then `pip install -e ".[transcribe,dev]"`.

## License

scribeio is free and open source under the
[MIT License](https://github.com/00apm/scribeio/blob/main/LICENSE): you
may use, change and share it, also commercially, as long as the licence
text stays with it.

WhisperX, ffmpeg and the speech models are installed separately and come
with their own licences.
