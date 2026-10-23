# Uninstall

Uninstalling scribeio removes the program, but not what it downloaded on
the way: the speech models, PyTorch's download, ffmpeg. Together they can
take 10 GB or more. This page lists each one, where it is, and how to
remove it.

Several of them are shared: other programs may use the same models,
Python or ffmpeg. Each step says so; skip a step when something else on
the computer still needs it.

## Before you start

```
scribe check
```

Under **Models** it lists the models on this computer, the size of each,
and the folders they are in. Note the folders: once scribeio is gone,
`scribe check` is gone too.

## 1. Remove scribeio

Installed with the [Quick install](install.md):

```
uv tool uninstall scribeio
```

This removes scribeio, WhisperX and PyTorch, and the `scribe` command. It
frees little space by itself: the large files stay in uv's download
cache (step 3).

Installed [into your own Python environment](install-own-env.md): delete
the environment if it was only for scribeio
(`conda env remove -n scribe`, or delete the venv folder). In an
environment you keep, `pip uninstall scribeio whisperx` removes scribeio
and WhisperX but leaves PyTorch and the other libraries they brought.

## 2. The speech models (about 2–6 GB)

Downloaded on the first transcription, into two folders in your home
folder (`C:\Users\you` on Windows):

| Folder | What scribeio put there |
|---|---|
| `.cache\huggingface\hub` | the Whisper models (`models--Systran--faster-whisper-large-v3` and similar) and most alignment models (`models--…wav2vec2…`) |
| `.cache\torch\hub\checkpoints` | the alignment models for English, French, German, Spanish and Italian (files starting with `wav2vec2_`) |

Delete those folders or files. Both folders are shared: other programs
that use Hugging Face or PyTorch models keep theirs there too, so delete
only the ones named above, not the whole folder. A different location
set with `HF_HOME` or `TORCH_HOME` is the one `scribe check` showed.

## 3. The download cache (a few GB)

The Quick install downloads PyTorch with its CUDA libraries, the biggest
part of the install, into uv's cache:

```
uv cache clean
```

This empties the whole cache. Other programs installed with uv keep
working; their next install or upgrade just downloads again.

Installed with pip or conda: `pip cache purge`, and `conda clean --all`
for conda's package cache.

## 4. The Python uv installed (about 100 MB)

The Quick install brought its own Python 3.12. If you installed no other
program with uv (`uv tool list` shows what is left):

```
uv python uninstall 3.12
```

## 5. ffmpeg (about 100–200 MB)

Other programs (video editors, OBS, media players) may use the same
ffmpeg. If none does:

```
winget uninstall --id Gyan.FFmpeg -e
```

macOS: `brew uninstall ffmpeg`. Linux: `sudo apt remove ffmpeg`, or your
distribution's package manager.

## 6. uv

If nothing else on the computer uses uv, see
[uninstalling uv](https://docs.astral.sh/uv/getting-started/installation/#uninstallation).

## What stays: your files

scribeio writes only into your session folders: `scribe.yaml`,
`glossary.txt`, the session audio and the transcripts, next to the
recordings. They are yours; nothing above touches them.
