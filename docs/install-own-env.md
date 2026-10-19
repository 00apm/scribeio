# Install into your own Python environment

Install scribe with pip into a venv, a conda env or a uv project, for
example next to a PyTorch you already have.

## What you need

| | Version |
|---|---|
| Python | 3.10–3.13 |
| ffmpeg | 6.1 or newer, with the mp3 encoder (libmp3lame); without it, `--format m4a` |
| WhisperX (transcription) | `>=3.8.6,<3.9`; 3.8.6 needs PyTorch 2.8 (`torch~=2.8.0`, `torchaudio~=2.8.0`, `torchvision~=0.23.0`) |
| GPU (transcription) | NVIDIA with CUDA; see [GPU](#gpu) |

Use an environment of its own: WhisperX pins its PyTorch version and would
replace the one other tools in a shared environment use.

## Install

> **Warning:** install PyTorch with CUDA *before* scribe. Otherwise pip
> takes PyTorch from PyPI, which on Windows is a CPU-only build, and
> transcription takes hours. The PyTorch version must also fit WhisperX's
> pin (2.8 for WhisperX 3.8.6): with a newer one, pip replaces it the same
> way.

1. PyTorch 2.8 with CUDA, from
   [pytorch.org's previous versions](https://pytorch.org/get-started/previous-versions/)
   (pick the CUDA build for your driver), for example CUDA 12.8:

   ```
   pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
   ```

2. scribe with transcription:

   ```
   pip install "scribeio[transcribe]"
   ```

3. ffmpeg on PATH: your OS package manager (see
   [Quick install, step 1](install.md#1-install-ffmpeg)) or conda-forge.

- **conda:** `conda create -n scribe python=3.12 ffmpeg -c conda-forge`,
  activate it, then steps 1 and 2. conda-forge ffmpeg builds differ:
  `scribe check` shows whether yours has the mp3 encoder.
- **uv:** in a uv-managed venv, `uv pip install "scribeio[transcribe]"
  --torch-backend=auto` picks the CUDA build itself; in a uv project, see
  uv's [PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
- **Windows PowerShell 5.1** may refuse to load the conda profile
  (execution policy): start it with `powershell -ExecutionPolicy Bypass`
  for one window, or run `Set-ExecutionPolicy RemoteSigned -Scope
  CurrentUser` once.

## Install one part only

- **Audio mastering only:** `pip install scribeio`, plus ffmpeg. No
  PyTorch, no WhisperX, no GPU.
- **Transcription only:** the full install; `scribe init --only
  transcribe` writes a config without the audio settings.

## Check the install

```
scribe check
```

It shows each use as ready or not, with versions, the GPU and the models
on disk; the output is explained on [Quick install](install.md#check-the-install).
The first line names the environment scribe runs in.

## Upgrade scribe

In the environment scribe is in:

```
pip install --upgrade "scribeio[transcribe]"
```

pip upgrades scribe and WhisperX and keeps your PyTorch while it fits
WhisperX's pin. Then run `scribe check`: if a newer WhisperX needed
another PyTorch, pip took it from PyPI, and on Windows that is a CPU-only
build; [PyTorch with CUDA](#pytorch-with-cuda) fixes it. Without
transcription: `pip install --upgrade scribeio`.

## Where the models go

- Speech models download on the first transcription into per-user caches
  (Hugging Face's, and torch's for some alignment models), not into the
  environment. Every install on the machine shares them.
- A custom `HF_HOME` set only inside one environment (for example as a
  conda env variable) moves the cache for that environment only; other
  installs download the models again.
- `transcribe.whisper.model` in `scribe.yaml` also accepts a folder
  holding a Whisper model in faster-whisper's format.

## Troubleshooting

Most problems are the same as for the quick install; the sections below
link there where they are.

### ffmpeg

See [Quick install: ffmpeg](install.md#ffmpeg); in a conda env,
`conda install ffmpeg -c conda-forge` also works.

### MP3 encoder

See [Quick install: MP3 encoder](install.md#mp3-encoder). In a conda env
the env's ffmpeg comes first on PATH: install a build with libmp3lame, or
use `--format m4a`.

### Transcription

"WhisperX not installed": `pip install "scribeio[transcribe]"` (after
PyTorch, see [Install](#install)). "installed but does not load": often a
PyTorch that does not fit WhisperX's pin; reinstall PyTorch 2.8 as in
step 1.

### PyTorch with CUDA

`scribe check` shows "PyTorch 2.8.0: a CPU-only build" (a CUDA build reads
"2.8.0+cu128, CUDA build"). Replace it:

```
pip uninstall -y torch torchaudio torchvision
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

Then `scribe check` again.

### GPU

See [Quick install: GPU](install.md#gpu): the driver, running on the CPU
(`transcribe.whisper.device: cpu`, very slow), and GPU memory.

### Connection problems

See [Quick install: connection problems](install.md#connection-problems).
