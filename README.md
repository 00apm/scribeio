# scribe

Transcripts of tabletop RPG sessions recorded with [Craig](https://craig.chat)
on Discord. One Craig `.zip` in → a transcript an LLM assistant can work with,
plus a quality report.

- Craig records one track per player, so no speaker diarization is needed:
  speaker names come from a small config.
- Whisper's hallucinations on silent stretches are removed by checking the
  player's own track for sound; every removal is listed in the report.
- Changed a name, the glossary or a threshold? `scribe merge` rebuilds the
  transcript in seconds, without the GPU.

scribe is a small app on top of [WhisperX](https://github.com/m-bain/whisperX);
transcription quality and speed come from WhisperX and the Whisper model.

## Install

Needs Python 3.10–3.13 and ffmpeg in `PATH`. An NVIDIA GPU is strongly
recommended: on a CPU, transcription takes hours per track.

The recommended way is to install WhisperX yourself, so you know what lands on
your disk:

1. **PyTorch with CUDA** (about 2–3 GB), from
   <https://pytorch.org/get-started/locally/>, for example:
   `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`
2. **WhisperX:** `pip install "whisperx>=3.8.6,<4"`
3. **scribe:** from a clone of this repository, `pip install .`

`pip install ".[transcribe]"` does steps 2 and 3 in one go. Without WhisperX,
`scribe merge` still works.

`scribe transcribe` needs two models: Whisper (`large-v3` by default, about
3 GB) and a word-alignment model for the language (about 0.4 GB for English,
1.2 GB for Polish). Models already on your disk are used offline. If one is
missing, scribe shows its size and asks before downloading it; for a missing
Whisper model it can also use another one you already have, for that run.
`--yes` downloads without asking. A cut-off download is detected and
replaced, never used.

Before any slow work, `scribe transcribe` checks that ffmpeg is in `PATH`,
WhisperX is installed and, with the default `device: cuda`, that torch is a
CUDA build and sees your GPU. The versions used are written into the `.json`
and the report.

## Usage

```
scribe init [RECORDING] [-o DIR] [--only master|transcribe] [--full] [--language LANG] [--force]
scribe transcribe INPUT [-c CONFIG] [-g GLOSSARY] [-o DIR] [--name NAME] [--force] [--keep-fillers] [--yes]
scribe merge JSON       [-c CONFIG] [-g GLOSSARY] [-o DIR] [--name NAME] [--keep-fillers]
```

`INPUT` is the Craig zip (download it as FLAC, multi-track), or a folder with
`info.txt` and `<n>-<account>.flac` tracks. `scribe <command> --help` lists
every option.

Settings live in one file, `scribe.yaml` (speakers, glossary, language,
everything else optional), plus `glossary.txt`. `scribe init` writes both from
commented templates; given a recording, it lists its Discord accounts so you
only fill in the names. `transcribe` and `merge` use `scribe.yaml` by themselves
when it sits next to the recording or the `.json`; `-c` points to another one.
Without any config, `transcribe` offers to create one, or continues with the
defaults (English, account names as speakers).

A typical session:

1. `scribe init "session.zip" --language pl`, then fill in the names in
   `scribe.yaml` (and campaign names in `glossary.txt`). Copy both to the next
   session's folder to reuse them.
2. `scribe transcribe "session.zip"` writes, next to the zip:
   - `2026-01-01_transcript.md`: the transcript, frontmatter plus `[01:02:13] Name: text` lines
   - `2026-01-01_transcript.json`: raw data (all segments, word timings, measurements)
   - `2026-01-01_transcript-report.md`: per-track summary, warnings, every dropped
     segment, possible bleed between tracks
3. Read the report. Add unknown accounts to `speakers:`, misheard names to the
   glossary.
4. `scribe merge 2026-01-01_transcript.json` rebuilds the `.md` and the report.
5. Give the `.md` to your assistant.

The templates, with every option explained:
[`scribe.yaml`](src/scribe_rpg/templates/scribe.yaml) and
[`glossary.txt`](src/scribe_rpg/templates/glossary.txt).

## Languages

Set `transcribe.whisper.language` in your config (default `en`). Any language
Whisper supports works if WhisperX has a word-alignment model for it (or you
set `transcribe.whisper.align_model`). English and Polish are tuned: the list
of known Whisper hallucination phrases and the hesitation filter (`uh`, `um`,
`erm` / `yyy`, `eee`, `mmm`) cover both. The main hallucination filter checks
for sound on the speaker's own track, so it works in every language.

## Supported setups

| | |
|---|---|
| Tested | WhisperX 3.8.6, model `large-v3`, NVIDIA GPU (CUDA, float16), English and Polish |
| Should work, untested | other Whisper models or a local model folder (`transcribe.whisper.model`); CPU (`device: cpu`, `compute_type: int8` or `float32`); WhisperX 3.x newer than 3.8.6; other languages with an alignment model |
| Required | WhisperX with word alignment (word timings drive the filters and the line building) |
| Not supported | other Whisper backends (`openai-whisper`, plain `faster-whisper`) |

Untested setups get a warning, never a block. The cleanup thresholds were tuned
on `large-v3`; smaller models hallucinate more, so check the report.

## Troubleshooting

**Transcription is much slower than expected (Windows).** When GPU memory is
full, the NVIDIA driver quietly moves the rest into normal RAM ("Shared GPU
memory" in Task Manager rises above zero) and work slows down several times,
without any error. scribe keeps only one model on the GPU at a time and warns
when little memory is left after loading Whisper. If it still happens, lower
`transcribe.whisper.batch_size`, or set *NVIDIA Control Panel → Manage 3D
settings → CUDA - Sysmem Fallback Policy* to *Prefer No Sysmem Fallback*: you
then get a clear out-of-memory error, which scribe explains.

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
