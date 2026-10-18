# Audio mastering

Turn a session recording into one evenly balanced audio file, and adjust
how it sounds.

## Configure the audio mastering

```
scribe init --only master "craig-session-7.zip"
```

The input is the Craig zip or a folder of FLAC tracks
([what scribe reads](../recordings.md)). This writes `scribe.yaml` next to
it: the file where your settings for this recording live (the mastering
steps you change, changes for one player, the file format). It holds the
settings people change most, and a comment listing the recording's
accounts. `--full` writes every setting with its default value.

scribe works without it, using the default settings (it says so), but
every change described below goes into this file. Every scribe command
uses it by itself when it sits next to the input. Without `--only master`,
the file also gets the speakers and the glossary for
[transcription](transcribe.md).

> **Warning:** `scribe.yaml` and the outputs land next to the input, not
> inside it. Run scribe from the session folder, with the tracks inside it
> ([keep the tracks inside a session folder](../recordings.md#keep-the-tracks-inside-a-session-folder)).

## Make the session audio

```
scribe master "craig-session-7.zip"
```

Every player's track goes through the same steps: low rumble and desk
thumps cut, every player brought to the same level, loud and quiet words
evened out, a little warmth and clarity added. The music bot is set well
under the voices. Then everything is mixed, peaks are capped, and one file
is written next to the input.

Every run starts by checking the settings, so a typo stops scribe within a
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
folder. If the file already exists, scribe stops; `--force` overwrites it.

## Preview your settings

```
scribe master "craig-session-7.zip" --preview
```

Instead of the whole session, scribe picks a 90-second excerpt and writes a
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

### How scribe picks the excerpt

scribe looks for a stretch in which every player talks, so you can check
how each person sounds. Of those stretches it takes the one where the
quietest player talks most (more music breaks a tie).

If no stretch of that length has every player, scribe makes a montage: an
equal piece for each player, taken from where that player talks most (90
seconds and four players give about 22 seconds each). It says so:

```
⚠ no stretch has every voice; montage of each voice's most active part
excerpt (montage): 00:01:36-00:01:58, 00:04:07-00:04:29, 00:08:47-00:09:29
```

Pieces that overlap are joined, so four players can give three parts.

A player who stays silent in the excerpt is named in a warning. With
`--solo`, only that player counts; with `--at`, scribe skips the search.

| Option | What it does |
|---|---|
| `--solo player.one` | one player alone, through their own chain, no mix |
| `--before-after` | only `00-original` and the final file |
| `--open` | opens the folder when done |
| `--at 01:15:00` | the excerpt starts here instead of where scribe picks |
| `--length 60` | excerpt length in seconds (default 90) |
| `--montage` | always makes the montage (not with `--at`) |

## Change a setting

Loudness is given in LUFS: loudness as the ear hears it, in negative
numbers; closer to 0 is louder, and a change of a few units is clearly
audible.

Each player's track goes through the `voice` chain, the music bot through
`music`, and the mixed result through `mix`. The steps, in order:

| Step | What it does | What you hear if you change it | Setting |
|---|---|---|---|
| `voice` `rumble` | cuts the lowest sounds: desk thumps, mic rumble | more cut: less boom, but voices get thinner | `f: 80` (Hz): higher cuts more, lower keeps more bass |
| `voice` `level` | brings every player to the same loudness | it sets how hard `comp` works | `target: -19`: usually leave it |
| `voice` `comp` | evens out loud and quiet words | stronger: steadier, but flatter and less lively | `ratio: 3`: higher evens out more, lower sounds more natural |
| `voice` `eq` | adds warmth (150 Hz) and clarity (3–5 kHz) | more clarity: crisper words, but harsher "s" sounds | the dB in `entry(3300,3);entry(5000,3)`: higher is brighter, 0 is off |
| `voice` `loudness` | the final voice level | louder or quieter voices; the music stays where it is | `target: -19`: closer to 0 is louder |
| `music` `loudness` | the music level | louder or quieter music under the voices | `target: -36`: closer to 0 is louder |
| `mix` `limiter` | caps the loudest peaks | rarely audible; lower caps more often | `limit: -1.5dB`: usually leave it |

Four more voice steps are ready but in no chain: `hum` (mains hum; set `f:
60` in the Americas), `denoise` (steady background hiss), `gate` (quieter
between words) and `deess` (softer sharp "s" sounds). They are untested
starting points; to use one, add its name to `chain`, e.g. `chain: [rumble,
hum, level, comp, eq, loudness]`. A list you write replaces the default
one, so write the whole chain.

Deeper settings (every option of every step) get a reference page after the
first release.

### Try a change and listen

1. In `scribe.yaml`, under `master:`, change one value; only the option you
   change needs writing, the step keeps its other settings:

   ```yaml
   master:
     voice:
       comp: {ratio: 4}
   ```

2. Run `scribe master "craig-session-7.zip" --preview` again.
3. Listen to `03-voice-comp.flac` against your copy from the last try, and
   to `08-final.mp3`.
4. Repeat until it sounds right, then run `scribe master` without
   `--preview` for the whole session.

### Change one player only

A player is one person's track; the voice chain is what all of them go
through. Under `players:`, a player's account gets changes over `voice`,
with the same keys:

```yaml
master:
  players:
    player.two: {chain: [rumble, level, comp, eq, denoise, loudness]}
    player.three: {eq: {gain_entry: "entry(80,0);entry(150,0);entry(3300,1);entry(5000,1);entry(6000,0)"}}
```

The account is written as Discord shows it (the comment `scribe init`
writes lists them). `--preview` says which players have their own chain;
`--solo player.two --preview` lets you hear one of them alone.

### Keep a stereo mic stereo

Voices are mixed in mono by default. For a player with a stereo mic:

```yaml
master:
  channels: stereo
  players:
    player.one: {stereo: true}
```

`channels: stereo` makes the session file stereo (the music keeps its
stereo); `stereo: true` keeps that player's track stereo too
(`voice: {stereo: true}` does it for everyone).

## Choose the file format

```
scribe master "craig-session-7.zip" --format m4a
```

| Format | Size per hour (default bitrate) | Plays on |
|---|---|---|
| `mp3` (default) | about 43 MB (96k) | everything |
| `m4a` | about 29 MB (64k) | phones and most players |
| `opus` | about 22 MB (48k) | browsers, VLC; not every older player |

`format:` in `scribe.yaml` sets it for every run. If ffmpeg has no MP3
encoder, use `m4a` ([MP3 encoder](../install.md#mp3-encoder)).

## Write each processed track

```
scribe master "craig-session-7.zip" --tracks
```

Besides the session file, this writes `2026-09-22_tracks/`: one FLAC per
track after its chain, without the mix, plus Craig's `info.txt`. The
folder is a valid scribe input itself: it can be
[transcribed](transcribe.md) or edited further in an audio editor.
