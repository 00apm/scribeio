# Your settings file

`scribe.yaml` holds scribeio's settings for one recording: who is who,
which language is spoken, how the audio should sound. This page explains
why you need it, how to read it, and how to change it without breaking
it.

## Why scribeio needs it

scribeio only sees Discord accounts and sounds. `scribe.yaml` tells it
what it cannot know by itself:

- **Who is who:** the name to write in the transcript for each Discord
  account (`gm.account` → `Game Master`).
- **Which track is music:** the music bot's name, so its track is mixed
  in as music and never transcribed (Kenku by default).
- **Which language is spoken:** English by default.

It also holds the settings that change scribeio's default behaviour:
how the audio is mastered (voice and music levels, one player's sound,
the file format) and how the transcript is made (the Whisper model, the
GPU or CPU, when a new line starts). They all work as they are; change
them when the result is not what you want.

scribeio runs without the file, too: it then uses the account names, the
default settings and English, and says so.

## Create the settings file

```
scribe init "craig-session-7.zip"
```

This writes `scribe.yaml` next to the recording, with the recording's
Discord accounts already listed, and `glossary.txt` for the names of your
world ([teach Whisper your names](usage/transcribe-tuning.md)). Every
`scribe` command uses `scribe.yaml` by itself when it sits next to the
input; it says so on its first line:

```
using config scribe.yaml (next to the input)
```

A file somewhere else is passed with `-c`:
`scribe master "craig-session-7.zip" -c "path/to/campaign.yaml"`.

`--only master` or `--only transcribe` writes only the settings that one
command reads.

> [!TIP]
> Playing with the same group (a campaign, a regular table)? When the
> players and your settings stay the same, copy `scribe.yaml` and
> `glossary.txt` from the last session's folder into the new one instead
> of running `scribe init`: the names, the glossary and your changes carry
> over. Add the account when someone new joins.

## How to read it

`scribe.yaml` is a [YAML](https://yaml.org) file: plain text, open it in
any text editor (Notepad, TextEdit, VS Code). A piece of it:

```yaml
master:                     # a section: the settings of `scribe master`
  channels: mono            # a setting: key, colon, space, value
  format: mp3               # two spaces in: it belongs to `master`
  # voice:                  # a line starting with # is not read:
  #   loudness: {target: -19}   # here it shows a default you can switch on
```

- **Sections** start at the left edge: `speakers` (names), `input`
  (reading the recording), `master` (the audio), `transcribe` (the
  transcript).
- **Indentation** says what belongs to what: two spaces per level, never
  tabs.
- **`#`** starts a note that scribeio ignores, to the end of the line.
- **`{target: -19}`** is a short way to write a setting with its own
  settings on one line; **`[kenku, rpgbot]`** is a list.

## Change a setting

A line starting with `#` shows a default or an example. To use it, remove
the `#` **and the space after it**, on every line of the block, and
change the value:

Before:

```yaml
master:
  channels: mono
  format: mp3
  # voice:                  # every voice track
  #   loudness: {target: -19}   # voice level; higher = louder
```

After (voices a little louder):

```yaml
master:
  channels: mono
  format: mp3
  voice:                    # every voice track
    loudness: {target: -17}   # voice level; higher = louder
```

**Save the file**, then run the command again: scribeio reads
`scribe.yaml` from the disk, so changes still open unsaved in your editor
do not count. Every run starts by checking the file, so a mistake stops
scribeio within a second, before any audio is processed.

## Common mistakes

The message names the file, often the line, and the setting.

| Mistake | What scribeio says | Fix |
|---|---|---|
| A tab instead of spaces | `scribe.yaml:2: invalid YAML (found character '\t' that cannot start any token)` | indent with spaces only |
| Only the `#` removed, so the line is one space too far in | `scribe.yaml:3: invalid YAML (mapping values are not allowed here)` | remove the space after `#` too |
| No space after the colon (`format:mp3`) | `scribe.yaml: master: expected key: value pairs` | `format: mp3` |
| A misspelt setting (`fromat`) | `scribe.yaml: master.fromat: unknown key` | spell it as in the [full file](#see-every-setting) |
| A name with `:` or `#` in it | `scribe.yaml:2: invalid YAML (mapping values are not allowed here)` | keep the quotes: `"gm.account": "Alice: the GM"` |
| A player block left empty | `scribe.yaml: master.players.player.one: expected key: value pairs` | uncomment its settings too |

## See every setting

`scribe.yaml` holds only the settings people change most. To see all of
them with their defaults and a note on each, write a full file into a
folder of its own:

```
scribe init "craig-session-7.zip" --full -o full
```

Copy what you need from `full/scribe.yaml` into your `scribe.yaml`. Only
copy what you change: a setting in your file wins over scribeio's
default, also when a later version improves that default.

> [!WARNING]
> Do not run `--full --force` next to your own `scribe.yaml`: it
> overwrites the file, names and changes included.

## Where each section is explained

- `speakers`: [transcribing a recording](usage/transcribe.md)
- `input`: [input recordings](recordings.md)
- `master`: [tuning the audio](usage/audio-tuning.md)
- `transcribe`: [tuning the transcription](usage/transcribe-tuning.md)
