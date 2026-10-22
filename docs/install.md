# Quick install

Install scribeio with three commands: ffmpeg, uv (it brings its own Python),
and scribeio itself.

> **What you need**
>
> - **For transcription, an NVIDIA GPU:** GTX 16xx, RTX 20xx or newer,
>   with at least 6 GB of memory; 8 GB recommended. Run `nvidia-smi`: a
>   table naming your GPU means the driver is ready. An older or smaller
>   GPU, no NVIDIA GPU, or a Mac: see [GPU](#gpu). Audio mastering needs
>   no GPU.
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
scribeio 0.1.3, Python 3.12.9 (installed with uv)

Audio mastering                                      ready
  ✔ ffmpeg 9.0.2  C:\Users\you\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe
  ✔ mp3 encoder

Transcription                                        ready
  ✔ WhisperX 3.8.6
  ✔ PyTorch 2.8.0+cu128, CUDA build
  ✔ GPU: NVIDIA GeForce RTX 4060, 8 GB
  ℹ ffmpeg 9: WhisperX's audio library prints a warning; harmless

Models (C:\Users\you\.cache\huggingface\hub)
  Whisper:   none yet: downloaded on your first transcription
  Alignment: none yet: downloaded on your first transcription
…
```

Each use is **ready**, **not ready**, or **not installed** (audio
mastering only). ✔ is fine, ℹ is for your information, ⚠ may cause
trouble, ✘ stops that use. Every ✘ comes with a link to the section below that fixes it:

```
Audio mastering                                      not ready
  ✔ ffmpeg 6.1  C:\Tools\ffmpeg\bin\ffmpeg.exe
  ✘ mp3 encoder: this ffmpeg has no libmp3lame, so it cannot write mp3 files; --format m4a works without it
    see https://github.com/00apm/scribeio/blob/main/docs/install.md#mp3-encoder
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

The speech models stay in the folder `scribe check` shows under Models.
Other programs that use Hugging Face models share that folder: delete the
models there only if nothing else needs them. To remove uv as well, see
[uninstalling uv](https://docs.astral.sh/uv/getting-started/installation/#uninstallation).

## Terminal tips

- **Open a terminal in a folder:** on Windows, right-click the folder →
  "Open in Terminal"; on macOS, right-click the folder → Services →
  "New Terminal at Folder". Paste a command and press Enter.
- **Put paths with spaces in quotes:** `"C:\RPG\Session 7"`, not
  `C:\RPG\Session 7`. Dragging a file onto the terminal pastes its path.
- **Learn the basics:** MDN's
  [command line crash course](https://developer.mozilla.org/en-US/docs/Learn_web_development/Getting_started/Environment_setup/Command_line).

## Troubleshooting

### `scribe` is not found

The folder uv puts programs in is not on your PATH yet:

```
uv tool update-shell
```

Then open a new terminal.

### ffmpeg

"not found in PATH": install it (step 1), then open a new terminal.
"older than 6.1": upgrade it (`winget upgrade --id Gyan.FFmpeg -e`,
`brew upgrade ffmpeg`); on an older Linux release, install a newer ffmpeg
build from [ffmpeg.org](https://ffmpeg.org/download.html).

### MP3 encoder

"this ffmpeg has no libmp3lame": the ffmpeg scribeio finds cannot write MP3
files. The ffmpeg from step 1 can; another ffmpeg that comes first in your
PATH (bundled with other software) may not. Either make the step 1 ffmpeg
the one found, or write M4A instead: `--format m4a` on the command, or in
`scribe.yaml`:

```yaml
master:
  format: m4a
```

### Transcription

"WhisperX not installed": scribeio was installed without transcription; run
step 3 again with `--force`. "installed but does not load": a download
went wrong; the same command rebuilds the install.

### PyTorch with CUDA

"a CPU-only build": `--torch-backend=auto` found no NVIDIA driver when
scribeio was installed, so it chose the PyTorch build without GPU support.
Install or update the NVIDIA driver (until `nvidia-smi` shows your GPU),
then run step 3 again with `--force`.

### GPU

"not seen": PyTorch cannot use the GPU. Check that `nvidia-smi` shows it;
update the NVIDIA driver if it does not.

"the installed PyTorch does not support this GPU": the GPU is older than
the installed PyTorch supports. With the install above, that is the GTX 900
and 1000 series and older. Transcribe on the CPU instead (below).

No NVIDIA GPU, or a Mac: transcription still runs on the CPU, very slowly
(hours for a session). Set it in `scribe.yaml`:

```yaml
transcribe:
  whisper:
    device: cpu
```

Audio mastering needs no GPU.

GPU memory: the default settings need up to about 6 GB, so 8 GB leaves
room for Windows and open apps. With 6 GB, set `model: large-v3-turbo`
(faster, about the same quality) or `compute_type: int8_float16` under
`transcribe: whisper:`; either needs about 4.5 GB. Less than 6 GB:
transcribe on the CPU (`device: cpu`, above).

### Connection problems

"not reachable": the speech models download from huggingface.co (some
alignment models from download.pytorch.org) on the first transcription.
A firewall, antivirus or proxy may be blocking Python. `scribe check
--online` tests the sites your settings need; allow Python through, or
try another network.

### Windows Smart App Control

"Windows Smart App Control blocked one of its files": Smart App Control
(Windows 11) lets a program load only files it trusts: signed by a known
publisher, or known to Microsoft as safe. Most libraries scribeio uses
for transcription (PyTorch, WhisperX and the ones they bring) are not
signed, so on some PCs Windows blocks one of them. scribeio cannot change
that; audio mastering works either way. It can start after scribeio
worked before: a new Windows 11 runs Smart App Control in evaluation mode
first and may switch it on by itself later.

Two ways out:

- **Turn Smart App Control off:** Windows Security → App & browser
  control → Smart App Control settings → Off. This lowers the protection
  of the whole PC, so it is the PC owner's decision. Recent Windows 11
  updates let you turn it back on later; older ones need Windows
  reinstalled for that.
- **Run scribeio in WSL** (Windows Subsystem for Linux, a Linux system
  inside Windows; Smart App Control does not check programs there):
  install it with `wsl --install`, then follow this page's Linux steps
  inside WSL. An NVIDIA GPU works there with the usual Windows driver.
