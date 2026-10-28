# Tuning the transcription

Make the transcript better: add the names of your world and corrections
for words that come out wrong, set the language, pick the speech model and where it runs, and choose when a
new line starts. Every setting goes into `scribe.yaml`, under
`transcribe:` ([your settings file](../settings.md)); transcribing itself
is on [transcribing a recording](transcribe.md).

## Add names and corrections

Whisper, the speech model, knows common words, not the names of your
world: "Strahd" may come out as "Strad", "Skaven" as "Scaven", or
differently each time. `glossary.txt`, written by `scribe init` next to
`scribe.yaml`, helps in two ways, one part each:

```
# 1. names to listen for, most important first
Strahd
Barovia
Cthulhu
Skaven

# 2. corrections: wrong -> right
Cthulu -> Cthulhu
Scaven* -> Skaven
Borovia* -> Barovia
```

**1. Names to listen for**, one per line. Before Whisper listens to a
track, scribeio gives it these names, so it is more likely to hear them
right in the first place. The space for them is limited
(`prompt_max_tokens`, 200 by default): put the most important first;
scribeio warns about names left out. They help the next
`scribe transcribe`, not a transcript already made.

**2. Corrections**, one per line with `->`. They fix the finished
transcript: whatever Whisper heard, the wrong spelling on the left is
replaced by the right one.

- `Cthulu -> Cthulhu` replaces a whole word or phrase, ignoring case.
- `Borovia* -> Barovia`: a `*` at the end of the left side replaces any
  word that starts with it and keeps the ending, so `Borovian` becomes
  `Barovian`. In languages where names change their endings (Polish
  `Strahda`, `Strahdem`), one such rule covers every form.

The right side of a correction also counts as a name to listen for. Any
line with `->` is a correction and any other line a name, wherever it is
in the file; `#` starts a comment, so the examples `scribe init` writes
are switched on by removing their `#`.

A glossary somewhere else is set with `glossary:` in `scribe.yaml` (a path
relative to `scribe.yaml`), so one glossary can serve a whole campaign.

### Fix a misspelt name

When the transcript has a name wrong:

1. Add a correction to `glossary.txt`, e.g. `Strad* -> Strahd`, and save
   the file.
2. Rebuild the transcript from the `.json`:

   ```
   scribe merge "2026-09-22_transcript.json"
   ```

This takes seconds and needs no GPU. `scribe merge` rebuilds the
transcript and the report, never the `.json`; it also picks up changed
names under `speakers:`. New prompt names help only the next
`scribe transcribe`.

## Choose the language

```yaml
transcribe:
  whisper:
    language: pl
```

`scribe init --language pl` writes it for you. Any language code Whisper
knows works (`en`, `de`, `fr`, …). English and Polish are tuned and
tested: for those, scribeio knows Whisper's typical made-up phrases and
the hesitation sounds. For another language, see
[hallucinated phrases](#remove-other-hallucinated-phrases).

## Choose the model and the device

```yaml
transcribe:
  whisper:
    model: large-v3-turbo
    device: cuda
```

- **`model`:** the Whisper speech model. `large-v3` (the default, 2.9 GB
  download) is the most accurate. `large-v3-turbo` is faster at about the
  same quality and needs less GPU memory. `medium` and `small` are
  smaller and faster again, but miss more words. A new model is
  downloaded on the first run that uses it
  ([download the models](transcribe.md#download-the-models-on-the-first-run)).
- **`device`:** `cuda` runs on an NVIDIA GPU (the default); `cpu` works
  on any computer, but very slowly: hours for a session. On the CPU, a
  smaller model shortens the wait.

Out of GPU memory, or a GPU with 6 GB or less: lower `batch_size`, or see
[GPU](../troubleshooting.md#gpu) for the settings that need less memory.

## Choose when a new line starts

```yaml
transcribe:
  lines:
    pause: 1.5
```

A speaker's words stay on one line until they pause this many seconds.
Lower: more, shorter lines, closer to the rhythm of the talk. Higher:
fewer, longer lines, a long speech in one piece. A short reply from
someone else always lands where it was said. `scribe merge` applies a
change without transcribing again.

## Advanced: hallucinations and hesitations

### Remove other hallucinated phrases

Whisper sometimes writes things nobody said, most often in near silence.
scribeio drops a line mostly made of words with no voice under them;
`phrases` is a backstop: a short line (6 words or fewer) that contains
one of them goes too. The built-in list holds Whisper's usual English
and Polish phrases ("thanks for watching", "napisy stworzone", …).

> [!WARNING]
> A `phrases` list in your `scribe.yaml` replaces the built-in one; it
> does not add to it. Copy the built-in list from the full file
> ([see every setting](../settings.md#see-every-setting)) and add your
> phrases below it.

```yaml
transcribe:
  cleanup:
    phrases:
      # the built-in list, copied from the full file
      - thanks for watching
      # …
      # yours
      - vielen dank fürs zuschauen
```

Each phrase matches part of a line, ignoring case. Check the report after
the next `scribe merge`: it lists every dropped line with its reason.

### Keep hesitations

Lines made only of hesitations ("um", "uh", "yyy", "mmm") are dropped.
`keep_fillers: true` under `transcribe: cleanup:` keeps them.
