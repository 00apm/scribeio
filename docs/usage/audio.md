# Audio mastering

Turn a session recording into one evenly balanced audio file, and hear
on a short preview how it will sound. Changing how it sounds is on
[tuning the audio](audio-tuning.md).

## Configure the audio mastering

```
scribe init "craig-session-7.zip"
```

The input is the Craig zip or a folder of FLAC tracks
([input recordings](../recordings.md)). This writes `scribe.yaml` next to
it, the file for your settings ([your settings file](../settings.md)).
If you only want the audio, `scribe init --only master` writes only the
mastering settings, and no glossary. scribeio also works without
`scribe.yaml`, on its default settings.

> **Warning:** `scribe.yaml` and the outputs land next to the input, not
> inside it. Run scribeio from the session folder, with the tracks inside it
> ([keep the tracks inside a session folder](../recordings.md#keep-the-tracks-inside-a-session-folder)).

## Make the session audio

```
scribe master "craig-session-7.zip"
```

Every player's track goes through the same steps: low rumble and desk
thumps cut, every player brought to the same level, loud and quiet words
evened out, a little warmth and clarity added. The music track is set well
under the voices. Then everything is mixed, peaks are capped, and one file
is written next to the input.

Every run starts by checking the settings, so a typo stops scribeio within a
second, before any audio is processed. The message names where the problem
is:

```
master.voice.comp (acompressor): Value 30.000000 for parameter 'ratio' out of range [1 - 20]
```

When all goes well, the run shows the config it uses and the chains, and
ends with the written file:

```
using config scribe.yaml (next to the input)
✔ config check (0.1s)
…
voice: rumble → level → comp → eq → loudness
music: loudness
mix: limiter
…
✔ render 2026-09-22_master.mp3 (5 tracks + mix) (10.2s)
master: C:\RPG\Session 7\2026-09-22_master.mp3
```

The name is the session date; `--name` changes it, `-o` writes to another
folder. If the file already exists, scribeio stops; `--force` overwrites it.

## Preview your settings

```
scribe master "craig-session-7.zip" --preview
```

Instead of the whole session, scribeio picks a 90-second excerpt and writes a
folder of short listening files, one after each step. When all goes well,
the run ends like this:

```
excerpt: 00:05:44, 90 s
…
preview level: -23.3 LUFS (lowered: the loudest peak stays at -1 dBFS)
✔ preview: write 9 files (3.1s)
preview: C:\RPG\Session 7\2026-09-22_preview
```

| File | What you hear |
|---|---|
| `00-original.flac` | every track as recorded, mixed, nothing changed |
| `01-voice-rumble.flac` … `05-voice-loudness.flac` | the voices after each voice step in turn, without the music |
| `06-music-loudness.flac` | the music joins at its level |
| `07-mix-limiter.flac` | the mix with its peaks capped |
| `08-final.mp3` | what the session file will sound like |
| `tracks/` | the excerpt of every track, unprocessed |

All files are played back at one common level, so you hear what a step
changes in the sound, not that it is louder. The numbers follow the chains,
so they shift when you add a step. Each `--preview` replaces the
folder: copy a file elsewhere to compare it with the next try.

### How scribeio picks the excerpt

scribeio looks for a stretch in which every player talks, so you can check
how each person sounds. Of those stretches it takes the one where the
quietest player talks most (more music breaks a tie).

If no stretch of that length has every player, scribeio makes a montage: an
equal piece for each player, taken from where that player talks most (90
seconds and four players give about 22 seconds each). It says so:

```
⚠ no stretch has every voice; montage of each voice's most active part
excerpt (montage): 00:01:36-00:01:58, 00:04:07-00:04:29, 00:08:47-00:09:29
```

Pieces that overlap are joined, so four players can give three parts.

A player who stays silent in the excerpt is named in a warning. With
`--solo`, only that player counts; with `--at`, scribeio skips the search.

| Option | What it does |
|---|---|
| `--solo player.one` | one player alone, through their own chain, no mix |
| `--before-after` | only `00-original` and the final file |
| `--open` | opens the folder when done |
| `--at 01:15:00` | the excerpt starts here instead of where scribeio picks |
| `--length 60` | excerpt length in seconds (default 90) |
| `--montage` | always makes the montage (not with `--at`) |

## Change how it sounds

The voice and music levels, one player's sound, stereo, the file format
and the processing steps are on [tuning the audio](audio-tuning.md).
For one run in another format:

```
scribe master "craig-session-7.zip" --format m4a
```

## Write each processed track

```
scribe master "craig-session-7.zip" --tracks
```

Besides the session file, this writes `2026-09-22_tracks/`: one FLAC per
track after its chain, without the mix, plus Craig's `info.txt`. The
folder is a valid scribeio input itself: it can be
[transcribed](transcribe.md) or edited further in an audio editor.
