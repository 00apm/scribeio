# Input recordings

scribeio works from separate tracks: one voice track per person, plus the
music bot's track if you had one. Most Craig multi-track downloads give
exactly that, zipped or unpacked; so does any folder of FLAC files with
one speaker per file. This page shows which downloads work, where scribeio
puts its files, and how to prepare tracks yourself.

## Give scribeio a Craig download

Every `scribe` command takes one input: the zip Craig gives you, or a folder
of its FLAC tracks.

```
scribe transcribe "craig-session-7.zip"
scribe transcribe tracks
```

- **Zip:** use it as downloaded; there is no need to unpack it.
- **Folder:** every `.flac` file in it is one track. Craig's `info.txt`
  is optional; without it, give the session start yourself:

  ```
  scribe master tracks --start "2026-09-22 20:00"
  ```

  The start (local time) names the outputs (`2026-09-22_master.mp3`) and
  dates the transcript. With `info.txt`, `--start` still wins if given.

Music bot tracks are recognised by their name (Kenku by default): they are
mixed into the session audio as music and never transcribed. Other bots go
in `scribe.yaml` under `input: music_patterns`.

## Choose the right Craig format

| Craig download | Works? |
|---|---|
| FLAC | yes |
| Audacity project | yes (the project file is ignored, the FLACs are read) |
| Adobe Audition project | yes (as Audacity) |
| AAC, Ogg, or any other non-FLAC format | no: scribeio stops with "no FLAC tracks; download the recording from Craig as FLAC or as an Audacity project" |

Zipped or unpacked makes no difference: an unpacked Audacity or Audition
download (a folder with `info.txt` and a `…_data` folder of FLACs) is read
like the zip.

## Keep the tracks inside a session folder

> **Warning:** scribeio writes `scribe.yaml` and every output *next to* the
> input, in the folder that holds it, not inside it. For a track folder
> that is its parent folder. Running `scribe init .` from inside the track
> folder puts `scribe.yaml` one level up, and the outputs follow it there.

Give each session its own folder, put the tracks in it, and run scribeio from
the session folder:

```
Session 7/            ← run scribe here; scribe.yaml and outputs land here
  tracks/             ← the FLACs (one per speaker)
```

```
cd "C:\RPG\Session 7"
scribe init tracks
scribe master tracks
```

The same holds for a zip: `scribe.yaml` and the outputs go next to it. Each
command finds `scribe.yaml` by itself when it sits next to the input; `-o`
writes the outputs to another folder.

## Edit the tracks yourself

You can trim noise or remove a track before running scribeio; a few rules
keep the result right.

- **Keep FLAC**, one file per speaker.
- **Keep every track's start.** The tracks line up because they all start
  at the same moment. Cutting the beginning of one track shifts that
  speaker against everyone else; silence a stretch instead of deleting it.
- **The file name is the speaker's account.** scribeio takes the name
  without Craig's leading number (`2-player_one.flac` → `player_one`);
  `speakers:` in `scribe.yaml` gives it a name (`player.one` and
  `player_one` count as the same account).
- **With `info.txt` in the folder, keep Craig's track names and every
  player's track.** In a folder of FLAC files, scribeio stops when a player
  `info.txt` lists has no file. In an unpacked Audacity or Audition download
  (tracks in the `*_data` folder) it warns and goes on without that player.
  To rename or drop tracks, remove `info.txt` and give `--start`.

## Not using Craig?

Any recording with one track per person works, if:

- each person is in their own FLAC file;
- all files start at the same moment and cover the same session;
- you give the start with `--start "YYYY-MM-DD HH:MM"`.

Convert other audio to FLAC with ffmpeg (scribeio needs it anyway):

```
ffmpeg -i "Player One.wav" "Player One.flac"
```

Here the account is the whole file name, `Player One`.

OBS records per source, not per person: one microphone or one Discord
output holds everyone at once, so it does not give one track per speaker.
