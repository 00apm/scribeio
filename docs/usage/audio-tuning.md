# Tuning the audio

Change how the session audio sounds: the voice and music levels, one
player's sound, stereo, the file format, and the processing steps
themselves. Every change goes into `scribe.yaml`, under `master:`
([your settings file](../settings.md)); making the audio is on
[audio mastering](audio.md).

## Change the voice and music levels

Loudness is given in LUFS: loudness as the ear hears it, in negative
numbers. Closer to 0 is louder, and a change of 2–3 units is clearly
audible. The voices end at -19 and the music at -36, well under them.

The new `scribe.yaml` has both as lines to switch on (remove the "# "):

```yaml
master:
  voice:                    # every voice track
    loudness: {target: -17}   # voice level; higher = louder
  music:                    # the music track
    loudness: {target: -33}   # music level; higher = louder
```

Only the option you change needs writing: the step keeps its other
settings.

## Try a change and listen

1. In `scribe.yaml`, under `master:`, change one value, and save the file.
2. Run `scribe master "craig-session-7.zip" --preview`
   ([preview your settings](audio.md#preview-your-settings)).
3. Listen to the file after the step you changed (`03-voice-comp.flac`
   for `comp`) against your copy from the last try, and to
   `08-final.mp3`.
4. Repeat until it sounds right, then run `scribe master` without
   `--preview` for the whole session.

## The default steps

Each player's track goes through the `voice` chain, the music track through
`music`, and the mixed result through `mix`. A chain is a list of steps
run in order; each step is one setting block with the same name.

```yaml
master:
  voice:
    chain: [rumble, level, comp, eq, loudness]
  music:
    chain: [loudness]
  mix:
    chain: [limiter]
```

| Step | What it does | What you hear if you change it | Setting |
|---|---|---|---|
| `voice` `rumble` | cuts the lowest sounds: desk thumps, mic rumble | more cut: less boom, but voices get thinner | `f: 80` (Hz): higher cuts more, lower keeps more bass |
| `voice` `level` | brings every player to the same loudness | it sets how hard `comp` works | `target: -19`: usually leave it |
| `voice` `comp` | evens out loud and quiet words | stronger: steadier, but flatter and less lively | `ratio: 3`: higher evens out more, lower sounds more natural |
| `voice` `eq` | adds warmth (150 Hz) and clarity (3–5 kHz) | more clarity: crisper words, but harsher "s" sounds | the dB in `entry(3300,3);entry(5000,3)`: higher is brighter, 0 is off |
| `voice` `loudness` | the final voice level | louder or quieter voices; the music stays where it is | `target: -19`: closer to 0 is louder |
| `music` `loudness` | the music level | louder or quieter music under the voices | `target: -36`: closer to 0 is louder |
| `mix` `limiter` | caps the loudest peaks | rarely audible; lower caps more often | `limit: -1.5dB`: usually leave it |

Why in this order:

1. **`rumble` first:** thumps and rumble are loud but carry no speech;
   cut first, they cannot set off the compressor.
2. **`level` before `comp`:** the compressor works from a fixed level
   (`threshold: -15dB`), so it treats every player the same only when
   they all arrive equally loud. A track that is mostly silence and table
   bumps is raised a lot here, its background noise too: give that player
   `denoise` or `gate` ([one player only](#change-one-player-only)).
3. **`eq` after `comp`:** the tone is shaped on the evened-out voice.
4. **`loudness` last:** `comp` and `eq` change the level; this sets the
   final one that goes into the mix.
5. **`limiter` on the mix:** voices and music together can peak higher
   than each alone; it caps them just under the maximum, without making
   the mix louder.

> [!CAUTION]
> The `eq` curve needs care: it is a list of points (Hz, dB), with straight
> lines between them; the right curve depends on the mic and the voice.
> Change one dB value at a time and listen with `--preview`.

## Change one player only

A player is one person's track; the `voice` chain is what all of them go
through. Under `players:`, a player's Discord account gets changes over
`voice`, with the same keys, laid over it one by one:

```yaml
master:
  players:
    player.one:
      loudness: {target: -17}           # this voice a little louder
    player.two:
      chain: [rumble, level, comp, eq, denoise, loudness]   # plus noise removal
```

The account is written as Discord shows it (the `speakers:` list has
them). `--preview` says which players have their own chain;
`--solo player.two --preview` lets you hear one of them alone.

## Keep a stereo mic stereo

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

## Choose the file format and bitrate

```yaml
master:
  format: m4a
  bitrate: 96k              # null = the format's default
```

| Format | Size per hour (default bitrate) | Plays on |
|---|---|---|
| `mp3` (default) | about 43 MB (96k) | everything |
| `m4a` | about 29 MB (64k) | phones and most players |
| `opus` | about 22 MB (48k) | browsers, VLC; not every older player |

A higher bitrate gives a bigger file; for speech, the defaults already
sound clean. `--format` on the command line overrides `format:` for one
run. If ffmpeg has no MP3 encoder, use `m4a`
([MP3 encoder](../troubleshooting.md#mp3-encoder)).

## The available steps

A step is one of these types. Each ffmpeg filter takes its own options,
passed to ffmpeg unchanged: the link is the full list.

In the default chains:

| Type | Step | What it does |
|---|---|---|
| [`highpass`](https://ffmpeg.org/ffmpeg-filters.html#highpass) | `rumble` | cuts everything below a frequency (`f`) |
| `loudness` | `level`, `loudness` | brings the track to a loudness (`target`, in LUFS); scribeio's own step: it measures the whole track, then turns it up or down by one fixed amount |
| [`acompressor`](https://ffmpeg.org/ffmpeg-filters.html#acompressor) | `comp` | turns loud parts down, so loud and quiet words come closer |
| [`firequalizer`](https://ffmpeg.org/ffmpeg-filters.html#firequalizer) | `eq` | raises or lowers frequencies along a curve of points |
| [`alimiter`](https://ffmpeg.org/ffmpeg-filters.html#alimiter) | `limiter` (mix) | caps peaks at a ceiling |

Ready steps, in no chain (untested starting points):

| Type | Step | What it does |
|---|---|---|
| [`bandreject`](https://ffmpeg.org/ffmpeg-filters.html#bandreject) | `hum` | cuts one narrow band: mains hum at 50 Hz (`f: 60` in the Americas) |
| [`afftdn`](https://ffmpeg.org/ffmpeg-filters.html#afftdn) | `denoise` | removes steady background hiss or fan noise |
| [`agate`](https://ffmpeg.org/ffmpeg-filters.html#agate) | `gate` | turns the track down between words |
| [`deesser`](https://ffmpeg.org/ffmpeg-filters.html#deesser) | `deess` | softens sharp "s" sounds |

Allowed, with no step yet (write your own):

| Type | What it does |
|---|---|
| [`lowpass`](https://ffmpeg.org/ffmpeg-filters.html#lowpass) | cuts everything above a frequency: hiss, harsh highs |
| [`equalizer`](https://ffmpeg.org/ffmpeg-filters.html#equalizer) | raises or lowers one band (`f`, `width_type`, `width`, `g`): simpler than `firequalizer` |
| [`volume`](https://ffmpeg.org/ffmpeg-filters.html#volume) | changes the level by a fixed amount (`volume: 3dB`) |

> [!NOTE]
> Only these types are allowed: each keeps the track's length, its
> channels and its timing, so the tracks stay in sync in the mix. Any
> other type stops with `unknown type`.

## Change a chain

**Add a ready step:** name it in the chain. A chain you write replaces the
default one, so write the whole list:

```yaml
master:
  voice:
    chain: [rumble, hum, level, comp, eq, loudness]
```

**Add a step of your own:** give it a new name, a `type` and its options,
and name it in a chain:

```yaml
master:
  voice:
    chain: [rumble, level, comp, eq, hiss, loudness]
    hiss: {type: lowpass, f: 9000}
```

**Change a step:** write only the options you change (`comp: {ratio: 4}`).
A step's `type` cannot change; for another filter, use a new step name.

Every run checks the chains first and stops with a message naming the
problem:

| What went wrong | What scribeio says |
|---|---|
| a step defined but in no chain | `scribe.yaml: master.voice.hiss: step is in no chain` |
| a chain names a step that does not exist | `scribe.yaml: master.voice.chain: unknown step 'hsis'` |
| a type that is not allowed | `scribe.yaml: master.voice.trim: unknown type 'silenceremove'` |
| a step's type changed | ``scribe.yaml: master.voice.eq: a tweak cannot change `type`; use a new step name`` |
| a wrong ffmpeg option value | `master.voice.comp (acompressor): Value 30.000000 for parameter 'ratio' out of range [1 - 20]` |
