# Rebuild the transcript

`scribe transcribe` keeps everything Whisper heard in
`2026-09-22_transcript.json`. `scribe merge` builds the transcript and the
report again from that file, with your current settings: in seconds,
without the GPU, without listening to the recording again.

```
scribe merge "2026-09-22_transcript.json"
```

The files are named by the session date (`2026-09-22` here); yours have
your session's date, or the name you gave with `--name`.

## When to use it

After you change, in `scribe.yaml` or `glossary.txt`:

- a name under `speakers:` (someone's name was wrong or missing);
- a correction in the glossary
  ([add names and corrections](transcribe-tuning.md#add-names-and-corrections));
- when a new line starts (`lines: pause`,
  [choose when a new line starts](transcribe-tuning.md#choose-when-a-new-line-starts));
- the removal of hallucinations and hesitations
  ([advanced](transcribe-tuning.md#advanced-hallucinations-and-hesitations)).

Save the file, then run `scribe merge`. A change that affects what Whisper
hears needs a new `scribe transcribe` instead: the names to listen for in
the glossary, the language, the model.

## What it writes

When all goes well:

```
using config scribe.yaml (next to the input)
✔ merge 2026-09-22_transcript.json (0.0s)
2026-09-22_transcript.md: 87 lines, 5 segments dropped, 0 possible bleed - see 2026-09-22_transcript-report.md
```

It overwrites `2026-09-22_transcript.md` and
`2026-09-22_transcript-report.md` without asking (copy them first to
compare), and never changes the `.json`: you can rebuild as often as you
like.

It reads `scribe.yaml` next to the `.json`, and the glossary that
`scribe.yaml` names (`glossary: glossary.txt`, as `scribe init` writes
it).

| Option | What it does |
|---|---|
| `-c scribe.yaml` | another settings file |
| `-g glossary.txt` | another glossary, instead of the one `scribe.yaml` names |
| `-o folder` | writes the transcript and the report to another folder |
| `--name session-7` | another name for the files (default: the session date) |
| `--keep-fillers` | keeps the hesitations ("um", "yyy") this time |
