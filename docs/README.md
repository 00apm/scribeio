# scribeio documentation

scribeio turns a recording with one track per person (a
[Craig](https://craig.chat) download from Discord) into one balanced
audio file and a transcript with everyone's name. It runs on your own
computer.

## How it works

```
Craig download ──> scribe init ──> scribe.yaml (names, language, your settings)
                                     │
                                     ├──> scribe master      ──> 2026-09-22_master.mp3
                                     └──> scribe transcribe  ──> 2026-09-22_transcript.md
```

The files are named by your session's date.

## First time here

1. Install scribeio: [quick install](install.md), or into
   [your own Python environment](install-own-env.md).
2. Run `scribe check`: it shows what works and what is missing.
3. Create the settings file and write the names in it:
   [create the settings file](settings.md#create-the-settings-file).
4. Make the session audio: [audio mastering](usage/audio.md).
5. Make the transcript: [transcribing a recording](usage/transcribe.md).

## The docs

### Install

- [Quick install](install.md): scribeio and ffmpeg with a few commands,
  then a check that everything works.
- [Install into your own Python environment](install-own-env.md): pip
  into a venv, a conda env or a uv project.
- [Troubleshooting](troubleshooting.md): what each ✘ of `scribe check`
  means and how to fix it.
- [Uninstall](uninstall.md): scribeio and what it downloaded.

### Use

- [Input recordings](recordings.md): which recordings work, where the
  files land, preparing tracks yourself.
- [Your settings file](settings.md): why `scribe.yaml` exists, how to
  read and change it.
- [Audio mastering](usage/audio.md): make the session audio; preview it
  on a short excerpt.
- [Transcribing a recording](usage/transcribe.md): make the transcript;
  read it and the report.
- [Rebuild the transcript](usage/merge.md): apply new names and
  corrections in seconds, without the GPU.

### Tune

- [Tuning the audio](usage/audio-tuning.md): levels, one player, stereo,
  format, the processing steps.
- [Tuning the transcription](usage/transcribe-tuning.md): names and
  corrections, language, model, line length.

## I want to…

### The audio

- [Hear how it will sound before processing the whole session](usage/audio.md#preview-your-settings)
- [Make the voices louder or the music quieter](usage/audio-tuning.md#change-the-voice-and-music-levels)
- [Fix one player who sounds wrong](usage/audio-tuning.md#change-one-player-only)
- [Get a smaller file, or another format](usage/audio-tuning.md#choose-the-file-format-and-bitrate)

### The transcript

- [Get the names of my world right](usage/transcribe-tuning.md#add-names-and-corrections)
- [Fix a name that came out wrong, in seconds](usage/transcribe-tuning.md#fix-a-misspelt-name)
- [Transcribe another language](usage/transcribe-tuning.md#choose-the-language)
- [Transcribe faster, or without an NVIDIA GPU](usage/transcribe-tuning.md#choose-the-model-and-the-device)
- [Get shorter or longer lines](usage/transcribe-tuning.md#choose-when-a-new-line-starts)

### The settings file

- [Change a setting in `scribe.yaml`](settings.md#change-a-setting)
- [See every setting there is](settings.md#see-every-setting)
- [Use the same settings for every session](settings.md#create-the-settings-file)

### Something else

- [Use a recording not made with Craig](recordings.md#not-using-craig)
- [Remove scribeio and what it downloaded](uninstall.md)

## Something went wrong

- **`scribe check` shows a ✘, or `scribe` is not found:**
  [troubleshooting](troubleshooting.md).
- **scribeio stops on `scribe.yaml`:** [common mistakes](settings.md#common-mistakes).
- **The transcription runs out of GPU memory, or is very slow:**
  [transcription troubleshooting](usage/transcribe.md#troubleshooting).
