# Troubleshooting

`scribe check` shows what is wrong; every ✘ links to its section here.
Where the fix depends on how you installed scribeio, a section has a line
for each: **Quick install** ([with uv](install.md)) and **own
environment** ([pip or conda](install-own-env.md)). The first line of
`scribe check` says which: "installed with uv", or the environment's
folder.

Problems during a transcription (GPU memory running out, a slow run):
see [Transcribe: troubleshooting](usage/transcribe.md#troubleshooting).

## `scribe` is not found

**Quick install:** the folder uv puts programs in is not on your PATH
yet:

```
uv tool update-shell
```

Then open a new terminal.

**Own environment:** activate the environment scribeio is installed in.

## ffmpeg

"not found in PATH": install it
([Quick install, step 1](install.md#1-install-ffmpeg)), then open a new
terminal; in a conda env, `conda install ffmpeg -c conda-forge` also
works. "older than 6.1": upgrade it (`winget upgrade --id Gyan.FFmpeg
-e`, `brew upgrade ffmpeg`); on an older Linux release, install a newer
ffmpeg build from [ffmpeg.org](https://ffmpeg.org/download.html).

## MP3 encoder

"this ffmpeg has no libmp3lame": the ffmpeg scribeio finds cannot write
MP3 files. The ffmpeg from [Quick install, step 1](install.md#1-install-ffmpeg)
can; another ffmpeg that comes first in your PATH (bundled with other
software) may not. In a conda env the env's ffmpeg comes first: some
conda-forge builds lack the encoder. Either make an ffmpeg with the
encoder the one found, or write M4A instead: `--format m4a` on the
command, or in `scribe.yaml`:

```yaml
master:
  format: m4a
```

## Transcription

"WhisperX not installed": scribeio was installed without transcription.
**Quick install:** run [step 3](install.md#3-install-scribeio) again with
`--force`. **Own environment:** `pip install "scribeio[transcribe]"`,
after PyTorch (see [Install](install-own-env.md#install)).

"installed but does not load". **Quick install:** a download went wrong;
the step 3 command with `--force` rebuilds the install. **Own
environment:** often a PyTorch that does not fit WhisperX's pin;
reinstall PyTorch 2.8 as in [step 1](install-own-env.md#install).

"Windows Smart App Control blocked one of its files": see
[Windows Smart App Control](#windows-smart-app-control).

## PyTorch with CUDA

"a CPU-only build": transcription would take hours (a CUDA build reads
"2.8.0+cu128, CUDA build").

**Quick install:** `--torch-backend=auto` found no NVIDIA driver when
scribeio was installed, so it chose the PyTorch build without GPU
support. Install or update the NVIDIA driver (until `nvidia-smi` shows
your GPU), then run [step 3](install.md#3-install-scribeio) again with
`--force`.

**Own environment:** pip took PyTorch from PyPI, which on Windows is a
CPU-only build. Replace it:

```
pip uninstall -y torch torchaudio torchvision
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

Then `scribe check` again.

## GPU

"not seen": PyTorch cannot use the GPU. Check that `nvidia-smi` shows it;
update the NVIDIA driver if it does not.

"the installed PyTorch does not support this GPU": the GPU is older than
the installed PyTorch supports. With the quick install, that is the GTX
900 and 1000 series and older. Transcribe on the CPU instead (below).

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

## Connection problems

"not reachable": the speech models download from huggingface.co (some
alignment models from download.pytorch.org) on the first transcription.
A firewall, antivirus or proxy may be blocking Python. `scribe check
--online` tests the sites your settings need; allow Python through, or
try another network.

## Windows Smart App Control

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
  install it with `wsl --install`, then follow the Linux steps of
  [Quick install](install.md) inside WSL. An NVIDIA GPU works there with
  the usual Windows driver.
