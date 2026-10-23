# Quick install

Install scribeio with three commands: ffmpeg, uv (it brings its own Python),
and scribeio itself.

> **What you need**
>
> - **For transcription, an NVIDIA GPU:** GTX 16xx, RTX 20xx or newer,
>   with at least 6 GB of memory; 8 GB recommended. Run `nvidia-smi`: a
>   table naming your GPU means the driver is ready. An older or smaller
>   GPU, no NVIDIA GPU, or a Mac: see [GPU](troubleshooting.md#gpu).
>   Audio mastering needs no GPU.
> - **Disk:** up to about 12 GB with transcription (the install and the
>   speech models); audio mastering alone needs well under 1 GB.

Every step is a command to paste into a terminal. Rarely use one? See
[Terminal tips](#terminal-tips).

## Install scribeio

### 1. Install ffmpeg

scribeio reads and writes audio with ffmpeg.

```
winget install --id Gyan.FFmpeg -e
```

macOS: `brew install ffmpeg` ([Homebrew](https://brew.sh)). Linux:
`sudo apt install ffmpeg`, or your distribution's package manager.

### 2. Install uv

uv installs Python programs, each with its own Python, and keeps them apart.

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS and Linux:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close the terminal and open a new one: the new commands are found only in
new terminals.

### 3. Install scribeio

```
uv tool install "scribeio[transcribe]" --python 3.12 --torch-backend=auto
```

This installs scribeio with transcription: Python 3.12, WhisperX, and the
PyTorch build that suits your NVIDIA driver (`--torch-backend=auto`). It
downloads a few GB and takes several minutes.

## Check the install

```
scribe check
```

It lists what each use needs and what it finds, offline:

```
scribeio 0.1.4, Python 3.12.9 (installed with uv)

Audio mastering                                      ready
  ✔ ffmpeg 9.0.2  C:\Users\you\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe
  ✔ mp3 encoder

Transcription                                        ready
  ✔ WhisperX 3.8.6
  ✔ PyTorch 2.8.0+cu128, CUDA build
  ✔ GPU: NVIDIA GeForce RTX 4060, 8 GB
  ℹ ffmpeg 9: WhisperX's audio library prints a warning; harmless

Models
  Whisper:   none yet: downloaded on your first transcription
  Alignment: none yet: downloaded on your first transcription

Settings
…
```

After the first transcription, Models lists each model with its size and
the folders they are in:

```
Models: 3.3 GB
  Whisper:   large-v3 (2.9 GB)
  Alignment: en (360 MB)
  Folders:   C:\Users\you\.cache\huggingface\hub
             C:\Users\you\.cache\torch\hub\checkpoints
```

Each use is **ready**, **not ready**, or **not installed** (audio
mastering only). ✔ is fine, ℹ is for your information, ⚠ may cause
trouble, ✘ stops that use. Every ✘ comes with a link to the
[Troubleshooting](troubleshooting.md) section that fixes it:

```
Audio mastering                                      not ready
  ✔ ffmpeg 6.1  C:\Tools\ffmpeg\bin\ffmpeg.exe
  ✘ mp3 encoder: this ffmpeg has no libmp3lame, so it cannot write mp3 files; --format m4a works without it
    see https://github.com/00apm/scribeio/blob/main/docs/troubleshooting.md#mp3-encoder
```

Run `scribe check` in a session folder and it also checks that folder's
`scribe.yaml` and lists the models it needs.

## Install scribeio with audio mastering only

Without transcription there is no WhisperX or PyTorch to download:

```
uv tool install scribeio --python 3.12
```

`scribe check` then shows Transcription as "not installed (audio mastering
only)". To add transcription later, run the full command from step 3 with
`--force`.

## Upgrade scribeio

```
uv tool upgrade scribeio
```

## Uninstall scribeio

```
uv tool uninstall scribeio
```

This leaves the speech models, uv's download cache and ffmpeg on the
computer: [Uninstall](uninstall.md) lists each one and how to remove it.

## Terminal tips

- **Open a terminal in a folder:** on Windows, right-click the folder →
  "Open in Terminal"; on macOS, right-click the folder → Services →
  "New Terminal at Folder". Paste a command and press Enter.
- **Put paths with spaces in quotes:** `"C:\RPG\Session 7"`, not
  `C:\RPG\Session 7`. Dragging a file onto the terminal pastes its path.
- **Learn the basics:** MDN's
  [command line crash course](https://developer.mozilla.org/en-US/docs/Learn_web_development/Getting_started/Environment_setup/Command_line).


## Troubleshooting

Something not working? `scribe check` links to the fix; all of them are
on [Troubleshooting](troubleshooting.md).
