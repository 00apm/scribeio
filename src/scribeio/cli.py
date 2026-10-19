"""Command-line interface: `scribe transcribe`, `merge`, `master`, `init`, `check`.

Commands stay thin: they read options, call the packages and report
progress. Errors scribe expects (`ScribeError`) are shown as one red line
with exit code 1; anything else is a bug and keeps its traceback.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import click

from scribeio import __version__, checks, craig, init
from scribeio.config import Config, load_config
from scribeio.errors import InstallError, ScribeError
from scribeio.progress import Progress
from scribeio.transcribe import (
    markdown,
    models,
    pipeline,
    report,
    session_json,
    whisper,
)
from scribeio.transcribe.glossary import Glossary, load_glossary
from scribeio.transcribe.session import RawSession, Segment, TrackInfo

if TYPE_CHECKING:
    # numpy at runtime only inside `master` / `transcribe`, never for `merge`
    from scribeio.master.config import MasterConfig
    from scribeio.master.excerpt import Excerpt

F = TypeVar("F", bound=Callable[..., Any])


class _Group(click.Group):
    """Click group that reports ScribeError in red, exit code 1.

    An installation error is followed by a pointer to ``scribe check``.
    """

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except ScribeError as e:
            click.secho(str(e), fg="red", err=True)
            if isinstance(e, InstallError):
                click.echo(checks.CHECK_HINT, err=True)
            ctx.exit(1)


def _common(f: F) -> F:
    """Options shared by `transcribe` and `merge`."""
    options = [
        click.option(
            "-c", "--config", "config_path",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="Config file (scribe.yaml): speakers, glossary, settings.",
        ),
        click.option(
            "-g", "--glossary", "glossary_path",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="Glossary file; wins over transcribe.glossary in the config.",
        ),
        click.option(
            "-o", "--output", "out_dir",
            type=click.Path(file_okay=False, path_type=Path),
            help="Output folder (default: next to the input).",
        ),
        click.option("--name", help="Output name (default: session date / JSON name)."),
        click.option(
            "--keep-fillers", is_flag=True, help="Keep hesitations (yyy, mmm, uh, um)."
        ),
    ]  # fmt: skip
    for option in reversed(options):
        f = option(f)
    return f


def _parse_start(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> datetime | None:
    """Parse ``--start`` as local time; bad text is a usage error (exit 2)."""
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M").astimezone()
    except ValueError:
        raise click.BadParameter(
            f'{value!r}: expected "YYYY-MM-DD HH:MM" (local time)'
        ) from None


def _start_option(f: F) -> F:
    """The ``--start`` option: session start for input without info.txt."""
    option = click.option(
        "--start",
        callback=_parse_start,
        metavar='"YYYY-MM-DD HH:MM"',
        help="Session start (local time); wins over info.txt. Needed for a "
        "folder without info.txt.",
    )
    return option(f)


@click.group(cls=_Group)
@click.version_option(__version__, prog_name="scribe")
def main() -> None:
    """Transcripts of TTRPG sessions recorded with Craig (Discord)."""
    if sys.platform == "win32":
        # the old Windows console prints colour codes as text unless told
        # otherwise; click stopped doing this in 8.5
        import colorama

        colorama.just_fix_windows_console()


@main.command()
@click.argument(
    "input_path", metavar="INPUT", type=click.Path(exists=True, path_type=Path)
)
@_common
@_start_option
@click.option("--force", is_flag=True, help="Overwrite existing outputs.")
@click.option(
    "-y", "--yes", is_flag=True, help="Download missing models without asking."
)
def transcribe(
    input_path: Path,
    config_path: Path | None,
    glossary_path: Path | None,
    out_dir: Path | None,
    name: str | None,
    keep_fillers: bool,
    start: datetime | None,
    force: bool,
    yes: bool,
) -> None:
    """Transcribe a Craig recording.

    INPUT is the Craig .zip, or a folder of FLAC tracks (one per speaker;
    without info.txt it needs --start). Writes <name>_transcript.md,
    <name>_transcript.json (raw data for `scribe merge`) and
    <name>_transcript-report.md (quality report); <name> is the session date
    unless --name gives one. Needs WhisperX (`scribe check` shows what is
    installed); an NVIDIA GPU is strongly recommended.
    """
    # Imported here: numpy is only needed for transcription, not for merge.
    from scribeio.transcribe import energy

    progress = Progress()
    folder = input_path.resolve().parent
    if config_path is None:
        config_path = _find_config(folder, progress)
    if config_path is None:
        language = load_config().settings["transcribe"]["whisper"]["language"]
        progress.warn(
            f"no config (-c, or {init.CONFIG_NAME} next to the recording): using "
            f"defaults - language {language}, account names as speakers"
        )
        if _interactive() and click.confirm(
            f"Create {init.CONFIG_NAME} for this recording now?", default=True, err=True
        ):
            _create_config(input_path, folder, None, False, progress)
            return
    cfg = load_config(config_path, glossary_path, keep_fillers)
    gloss = _glossary(cfg)
    wcfg = cfg.settings["transcribe"]["whisper"]

    # The ExitStack keeps the zip's temp folder alive until outputs are written
    # and removes it afterwards, also on errors and Ctrl+C. Known noisy library
    # warnings go to the report instead of the screen; on a failure they are
    # shown, as they may explain it.
    with ExitStack() as stack:
        hidden = stack.enter_context(whisper.hidden_warnings())
        stack.push(lambda kind, *_: _show_hidden_on_error(kind, hidden, progress))

        # Cheap checks first: fail in seconds, before unzipping or loading models.
        with progress.step("check ffmpeg, WhisperX, CUDA"):
            run_warnings = whisper.preflight(wcfg)
            used_versions = whisper.versions()
        for w in run_warnings:
            progress.warn(w)
        with progress.step("check models"):
            needed = models.needed(wcfg)
        wcfg = _ensure_models(needed, wcfg, progress, yes)

        with progress.step(f"open {input_path.name}"):
            patterns = cfg.settings["input"]["music_patterns"]
            rec = stack.enter_context(craig.open_recording(input_path, patterns, start))
        paths = _paths(
            out_dir or input_path.resolve().parent,
            name or rec.start.strftime("%Y-%m-%d"),
        )
        existing = [p.name for p in paths.values() if p.exists()]
        if existing and not force:
            raise ScribeError(
                f"output already exists: {existing[0]} (use --force to overwrite)"
            )
        _print_music(rec, "not transcribed", progress)
        for t in rec.players:
            if craig.sanitize(t.account) not in cfg.speakers:
                progress.warn(
                    f"unknown account {t.account}: "
                    "no speakers: entry, account name used"
                )

        label = f"load Whisper {wcfg['model']} ({wcfg['language']}) on {wcfg['device']}"
        with progress.step(label):
            transcriber = whisper.Transcriber(wcfg, gloss.names)
        for w in transcriber.warnings:
            progress.warn(w)
        run_warnings += transcriber.warnings
        if transcriber.prompt_cut:
            cut = ", ".join(transcriber.prompt_cut)
            progress.warn(f"glossary names left out of the prompt: {cut}")

        cleanup = cfg.settings["transcribe"]["cleanup"]
        segments: list[Segment] = []
        duration = _music_duration(rec, progress, run_warnings)
        with progress.tracks(len(rec.players)) as bar:
            for t in rec.players:
                bar.start(t.account)
                audio = whisper.load_audio(t.path)
                duration = max(duration, len(audio) / energy.SAMPLE_RATE)
                found = transcriber.transcribe(
                    audio, t.account, on_progress=bar.update, on_phase=bar.set_phase
                )
                energy.annotate(found, audio, cleanup["energy_db"], cleanup["frame_ms"])
                segments.extend(found)
                bar.done(f"{t.account}: {len(found)} segments")
        estimated = sum(w.estimated for s in segments for w in s.words)
        if estimated:
            progress.warn(
                f"{estimated} words had no timestamp from alignment; times estimated"
            )

        raw = RawSession(
            date=rec.start.strftime("%Y-%m-%d"),
            start=rec.start.isoformat(timespec="seconds"),
            duration=round(duration, 3),
            source=rec.source,
            whisper=transcriber.settings(),
            tracks=[
                TrackInfo(t.account, t.kind, t.path.name, t.kind == "player")
                for t in rec.tracks
            ],
            segments=segments,
            scribe=__version__,
            versions=used_versions,
            warnings=[*run_warnings, *hidden],
        )
        with progress.step(f"write {paths['md'].name}"):
            paths["md"].parent.mkdir(parents=True, exist_ok=True)
            session_json.write(paths["json"], raw)
            result = _merge_and_write(raw, cfg, gloss, paths)
    _summary(progress, paths, result)
    if hidden:
        s = "" if len(hidden) == 1 else "s"
        progress.console.print(
            f"{len(hidden)} library warning{s} hidden, listed in the report",
            markup=False,
        )


def _show_hidden_on_error(
    kind: type[BaseException] | None, hidden: list[str], progress: Progress
) -> bool:
    """Print the hidden library warnings if the run stops on an error.

    Not on success, not on Ctrl+C. Returns False: the error goes on.
    """
    if hidden and kind is not None and issubclass(kind, Exception):
        progress.console.print("library warnings (may explain the error):")
        for w in hidden:
            progress.console.print(f"  {w}", markup=False)
    return False


def _print_music(rec: craig.Recording, use: str, progress: Progress) -> None:
    """Name the tracks `music_patterns` took for music bots, if any.

    A player matched by a pattern by mistake is then visible at once.
    """
    music = [t.account for t in rec.tracks if t.kind == "music"]
    if music:
        line = f"music ({use}): {', '.join(music)}"
        progress.console.print(line, markup=False, soft_wrap=True)


def _music_duration(
    rec: craig.Recording, progress: Progress, run_warnings: list[str]
) -> float:
    """Length of the longest music track, read before any transcription.

    Music bots often join first and leave last, so they count for the
    session's duration. A music track ffmpeg cannot read is only a warning
    (added to `run_warnings`): it is never transcribed.
    """
    from scribeio import ffmpeg

    longest = 0.0
    for t in rec.tracks:
        if t.kind == "music":
            try:
                longest = max(longest, ffmpeg.duration(t.path))
            except ScribeError as e:
                warning = f"{t.account}: length unknown, not in the duration ({e})"
                progress.warn(warning)
                run_warnings.append(warning)
    return longest


@main.command(short_help="Rebuild transcript and report from a .json (no GPU).")
@click.argument(
    "json_path",
    metavar="JSON",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_common
def merge(
    json_path: Path,
    config_path: Path | None,
    glossary_path: Path | None,
    out_dir: Path | None,
    name: str | None,
    keep_fillers: bool,
) -> None:
    """Rebuild transcript and report from a .json, without the GPU.

    JSON is a <name>_transcript.json from `scribe transcribe`. Use it after
    changing speakers, glossary or cleanup settings. Overwrites
    <name>_transcript.md and <name>_transcript-report.md; never changes the
    .json.
    """
    progress = Progress()
    if config_path is None:
        config_path = _find_config(json_path.resolve().parent, progress)
    if config_path is None:
        progress.warn(
            f"no config (-c, or {init.CONFIG_NAME} next to the .json): account "
            f"names used as speakers; create one with: scribe init {json_path.name}"
        )
    cfg = load_config(config_path, glossary_path, keep_fillers)
    gloss = _glossary(cfg)
    raw = session_json.read(json_path)
    paths = _paths(out_dir or json_path.resolve().parent, name or _json_name(json_path))
    with progress.step(f"merge {json_path.name}"):
        paths["md"].parent.mkdir(parents=True, exist_ok=True)
        result = _merge_and_write(raw, cfg, gloss, paths)
    for w in result.warnings:
        progress.warn(w)
    _summary(progress, paths, result)


def _ensure_models(
    needed: list[models.Model],
    wcfg: dict[str, Any],
    progress: Progress,
    yes: bool,
) -> dict[str, Any]:
    """Make sure every needed model is on disk, asking before any download.

    For a missing Whisper model the user may also pick another downloaded
    one for this run; scribe never switches on its own.

    Returns:
        The whisper settings to use (the model may have been switched).

    Raises:
        ScribeError: If the user aborts.
    """
    for model in needed:
        if model.cached:
            continue
        text = models.describe(model, models.size(model))
        progress.warn(text if model.damaged else f"{text} is not downloaded")
        others = models.cached_whisper_models() if model.kind == "Whisper model" else []
        if yes:
            choice = "d"
        else:
            options = {"d": "download it"}
            if others:
                options["u"] = f"use {others[0]} for this run"
            options["a"] = "abort"
            for key, meaning in options.items():
                progress.console.print(f"  [{key}] {meaning}", markup=False)
            choice = click.prompt(
                "Choose", type=click.Choice(list(options)), default="a", err=True
            )
        if choice == "a":
            raise ScribeError(
                f"{models.label(model)} not downloaded; run again and choose "
                "download, or add --yes"
            )
        if choice == "u":
            wcfg = {**wcfg, "model": others[0]}
            continue
        models.download(model, progress)
    return wcfg


AT = re.compile(r"(\d+):([0-5]\d):([0-5]\d)")


def _parse_at(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> int | None:
    """Parse ``--at HH:MM:SS`` to seconds; bad text is a usage error."""
    if value is None:
        return None
    m = AT.fullmatch(value)
    if m is None:
        raise click.BadParameter(f"{value!r}: expected HH:MM:SS")
    h, mnt, sec = map(int, m.groups())
    return h * 3600 + mnt * 60 + sec


@main.command()
@click.argument(
    "input_path", metavar="INPUT", type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Config file (scribe.yaml): speakers, master settings.",
)
@click.option(
    "-o",
    "--output",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output folder (default: next to the input).",
)
@click.option("--name", help="Output name (default: session date).")
@_start_option
@click.option("--force", is_flag=True, help="Overwrite existing outputs.")
@click.option(
    "--preview",
    is_flag=True,
    help="Write <name>_preview/ for a short excerpt instead of the full master.",
)
@click.option(
    "--at",
    callback=_parse_at,
    metavar="HH:MM:SS",
    help="Preview: excerpt start (default: found automatically).",
)
@click.option(
    "--length",
    type=click.IntRange(min=1),
    metavar="SECONDS",
    help="Preview: excerpt length (default 90).",
)
@click.option(
    "--montage", is_flag=True,
    help="Preview: join each voice's most active stretch.",
)  # fmt: skip
@click.option(
    "--before-after", is_flag=True,
    help="Preview: only 00-original and the final file.",
)  # fmt: skip
@click.option(
    "--solo", metavar="ACCOUNT",
    help="Preview: one voice track alone, through its own chain, no mix.",
)  # fmt: skip
@click.option(
    "--open", "open_after", is_flag=True,
    help="Preview: open the folder in the file manager when done.",
)  # fmt: skip
@click.option(
    "--format", "fmt", type=click.Choice(["mp3", "m4a", "opus"]),
    help="Output format (default: master.format in the config, else mp3).",
)  # fmt: skip
@click.option(
    "--tracks", "write_tracks", is_flag=True,
    help="Also write each processed track to <name>_tracks/ (FLAC).",
)  # fmt: skip
@click.option(
    "--verbose", is_flag=True,
    help="Print each track's measured loudness and applied gain.",
)  # fmt: skip
def master(
    input_path: Path,
    config_path: Path | None,
    out_dir: Path | None,
    name: str | None,
    start: datetime | None,
    force: bool,
    preview: bool,
    at: int | None,
    length: int | None,
    montage: bool,
    before_after: bool,
    solo: str | None,
    open_after: bool,
    fmt: str | None,
    write_tracks: bool,
    verbose: bool,
) -> None:
    """Mix and master a Craig recording into one audio file.

    INPUT is the Craig .zip, or a folder of FLAC tracks (one per speaker;
    without info.txt it needs --start). Writes <name>_master.<format>: every
    track through its chain, mixed, the mix chain, encoded. With --preview,
    writes only <name>_preview/ for a short excerpt: tracks/ (the excerpt of
    every track, unprocessed) and a listening file per step, at one loudness.
    """
    valued = [("--at", at), ("--length", length), ("--solo", solo)]
    used = [opt for opt, v in valued if v is not None]
    flags = [("--montage", montage), ("--before-after", before_after),
             ("--open", open_after)]  # fmt: skip
    used += [opt for opt, on in flags if on]
    if used and not preview:
        raise click.UsageError(f"{', '.join(used)} needs --preview")
    if at is not None and montage:
        raise click.UsageError("--at and --montage cannot be used together")
    if write_tracks and preview:
        raise click.UsageError("--tracks cannot be used with --preview")

    from scribeio.master import mix
    from scribeio.master.config import resolve, unknown_players

    progress = Progress()
    if config_path is None:
        config_path = _find_config(input_path.resolve().parent, progress)
    if config_path is None:
        progress.warn(
            f"no config (-c, or {init.CONFIG_NAME} next to the recording): using "
            "the default mastering settings"
        )
    cfg = load_config(config_path)
    section = cfg.settings["master"]
    # --format wins; the bitrate default then follows it unless the config set one
    mcfg = resolve({**section, "format": fmt} if fmt else section)
    found = checks.require(checks.ffmpeg_item())
    if found.status == "warning":
        progress.warn(checks.line(found))
    with progress.step("config check"):
        mix.check(mcfg)
    with ExitStack() as stack:
        with progress.step(f"open {input_path.name}"):
            patterns = cfg.settings["input"]["music_patterns"]
            rec = stack.enter_context(craig.open_recording(input_path, patterns, start))
        _print_music(rec, "music chain", progress)
        for t in rec.players:
            if craig.sanitize(t.account) not in cfg.speakers:
                progress.warn(
                    f"unknown account {t.account}: no speakers: entry, voice chain used"
                )
        for key in unknown_players(mcfg, [t.account for t in rec.players]):
            progress.warn(f"players: {key}: not in this recording, ignored")
        name = name or rec.start.strftime("%Y-%m-%d")
        folder = out_dir or input_path.resolve().parent
        if preview or write_tracks:
            # the folder is replaced, so it must not be (or hold) the input
            kind = "preview" if preview else "tracks"
            _check_not_input(folder / f"{name}_{kind}", input_path)
        if preview:
            account = _solo_account(rec, solo) if solo is not None else None
            clean = cfg.settings["transcribe"]["cleanup"]
            ex = _find_excerpt(rec, clean, length or 90, at, montage, account,
                               progress)  # fmt: skip
            out = folder / f"{name}_preview"
            _write_preview(rec, ex, mcfg, out, before_after, verbose, account,
                           progress)  # fmt: skip
            if open_after:
                _open(out, progress)
        else:
            tracks_dir = folder / f"{name}_tracks" if write_tracks else None
            _write_master(rec, mcfg, folder / f"{name}_master.{mcfg.format}",
                          tracks_dir, force, verbose, progress)  # fmt: skip


def _find_excerpt(
    rec: craig.Recording,
    clean: dict[str, Any],
    length: int,
    at: int | None,
    montage: bool,
    solo: str | None,
    progress: Progress,
) -> Excerpt:
    """Measure every track's activity and choose the preview excerpt.

    With `solo` (a sanitized account), only that voice counts: the window is
    where it is most active (music still breaks ties).
    """
    from scribeio.master import excerpt

    tracks = [
        t
        for t in rec.tracks
        if solo is None or t.kind == "music" or craig.sanitize(t.account) == solo
    ]
    with progress.step("find excerpt"):
        acts = excerpt.activities(
            [t.path for t in tracks], clean["frame_ms"], clean["energy_db"]
        )
        pairs = list(zip(tracks, acts, strict=True))
        voice = {t.account: a for t, a in pairs if t.kind == "player"}
        music = [a for t, a in pairs if t.kind == "music"]
        ex = excerpt.choose(voice, music, length, at, montage)
    _show_excerpt(ex, montage, progress)
    return ex


def _write_preview(
    rec: craig.Recording,
    ex: Excerpt,
    mcfg: MasterConfig,
    folder: Path,
    before_after: bool,
    verbose: bool,
    solo: str | None,
    progress: Progress,
) -> None:
    """Cut the excerpt out of every track, then write the listening files."""
    from scribeio.master import preview

    with progress.step(f"cut {len(rec.tracks)} tracks"):
        preview.fresh_folder(folder)
        paths = preview.cut(rec.tracks, ex.ranges, folder / "tracks")
    cut = [
        craig.Track(t.account, t.kind, p)
        for t, p in zip(rec.tracks, paths, strict=True)
    ]
    preview.write_files(cut, mcfg, folder, progress, verbose, before_after, solo)
    progress.console.print(f"preview: {folder}", markup=False, soft_wrap=True)


def _check_not_input(target: Path, input_path: Path) -> None:
    """Refuse an output folder that is, or holds, the input.

    Raises:
        ScribeError: If writing `target` would delete the input.
    """
    if input_path.resolve().is_relative_to(target.resolve()):
        raise ScribeError(
            f"output folder {target.name} holds the input: use --name or -o "
            "to write elsewhere"
        )


def _solo_account(rec: craig.Recording, solo: str) -> str:
    """The sanitized account `--solo` names among the voice tracks.

    Raises:
        ScribeError: If no voice track matches; lists the voice tracks.
    """
    wanted = craig.sanitize(solo)
    accounts = [craig.sanitize(t.account) for t in rec.players]
    if wanted not in accounts:
        names = ", ".join(t.account for t in rec.players)
        raise ScribeError(f"--solo: no voice track {solo!r}; voice tracks: {names}")
    return wanted


def _open(folder: Path, progress: Progress) -> None:
    """Open the preview folder; a failure is only a warning."""
    from scribeio.master import preview

    try:
        preview.open_folder(folder)
    except ScribeError as e:
        progress.warn(str(e))


def _print_chains(rec: craig.Recording, mcfg: MasterConfig, progress: Progress) -> None:
    """List the steps of every chain this run uses, one line per stage.

    A pass runs all its steps at once, so the progress bars cannot show them.
    Players with their own block get their own line.
    """
    from scribeio.master.catalogue import Step

    def show(label: str, steps: list[Step]) -> None:
        names = " → ".join(s.name for s in steps) or "(none)"
        progress.console.print(f"{label}: {names}", markup=False, soft_wrap=True)

    show("voice", mcfg.voice)
    for t in rec.players:
        account = craig.sanitize(t.account)
        if account in mcfg.players:
            show(account, mcfg.players[account])
    if any(t.kind == "music" for t in rec.tracks):
        show("music", mcfg.music)
    show("mix", mcfg.mix)


def _write_master(
    rec: craig.Recording,
    mcfg: MasterConfig,
    out: Path,
    tracks_dir: Path | None,
    force: bool,
    verbose: bool,
    progress: Progress,
) -> None:
    """Measure the loudness steps, then render the whole recording to `out`.

    With `tracks_dir`, the same run writes each processed track there, plus
    the recording's info.txt, so the folder is a valid input itself.
    """
    from scribeio.master import measuring, mix

    for path in [out, tracks_dir]:
        if path is not None and path.exists() and not force:
            raise ScribeError(
                f"output already exists: {path.name} (use --force to overwrite)"
            )
    _print_chains(rec, mcfg, progress)
    plans = [
        mix.TrackPlan(t, mcfg.chain_for(t), stereo=mcfg.keeps_stereo(t))
        for t in rec.tracks
    ]
    mix_gains = measuring.measure(plans, mcfg.mix, mcfg.channels, progress, verbose)
    out.parent.mkdir(parents=True, exist_ok=True)
    if tracks_dir is not None:
        part = mix.partial_path(tracks_dir)
        mix.discard(part)  # a leftover of a run that was stopped hard
        part.mkdir()
        if rec.info_txt is not None:
            (part / "info.txt").write_text(rec.info_txt, encoding="utf-8")
    n = len(plans)
    with progress.bar(
        f"render {out.name} ({n} track{'s' if n != 1 else ''} + mix)"
    ) as update:
        mix.render(plans, mcfg.mix, mix_gains, mcfg.channels, out, mcfg.format,
                   mcfg.bitrate, update, tracks_dir=tracks_dir)  # fmt: skip
    progress.console.print(f"master: {out}", markup=False, soft_wrap=True)
    if tracks_dir is not None:
        progress.console.print(f"tracks: {tracks_dir}", markup=False, soft_wrap=True)


def _show_excerpt(ex: Excerpt, forced: bool, progress: Progress) -> None:
    """Print the chosen excerpt, the montage notice and missing voices."""
    from scribeio.master.excerpt import hms

    if ex.montage:
        spans = ", ".join(f"{hms(s)}-{hms(e)}" for s, e in ex.ranges)
        progress.console.print(f"excerpt (montage): {spans}", markup=False)
        if not forced:
            progress.warn(
                "no stretch has every voice; montage of each voice's most active part"
            )
    else:
        [(s, e)] = ex.ranges
        progress.console.print(f"excerpt: {hms(s)}, {e - s} s", markup=False)
    if ex.missing:
        progress.warn(f"silent in the excerpt: {', '.join(ex.missing)}")


@main.command(name="init")
@click.argument(
    "source",
    metavar="[RECORDING]",
    required=False,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Folder for the files (default: next to RECORDING, else here).",
)
@click.option(
    "--only",
    type=click.Choice(["master", "transcribe"]),
    help="Only this command's settings (default: both; master needs no glossary).",
)
@click.option("--full", is_flag=True, help="Every setting with its default value.")
@click.option("--language", help="Transcription language, e.g. pl (default: en).")
@click.option("--force", is_flag=True, help="Overwrite an existing scribe.yaml.")
def init_command(
    source: Path | None,
    out_dir: Path | None,
    only: init.Only,
    full: bool,
    language: str | None,
    force: bool,
) -> None:
    """Create scribe.yaml and glossary.txt to fill in.

    The config is short: the settings people change. With RECORDING (a Craig
    .zip, a track folder or a <name>_transcript.json), the speakers section lists its
    Discord accounts, ready for their names. Every scribe command uses
    scribe.yaml by itself when it is next to the recording or the .json.
    """
    if language and only == "master":
        raise click.UsageError("--language is for transcription, not --only master")
    progress = Progress()
    if out_dir is None:
        out_dir = source.resolve().parent if source else Path.cwd()
    _create_config(source, out_dir, language, force, progress, only, full)


@main.command(short_help="Show what scribe needs and what it finds (offline).")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Config file to check (default: scribe.yaml in this folder).",
)
@click.option(
    "--online", is_flag=True,
    help="Also look up missing model sizes and test the model sources (network).",
)  # fmt: skip
@click.pass_context
def check(ctx: click.Context, config_path: Path | None, online: bool) -> None:
    """Show what scribe needs and what it finds, per use.

    Audio mastering (ffmpeg, the encoder), transcription (WhisperX, PyTorch,
    the GPU) and the models on disk, with a link to the guide for anything
    missing. Exits with 1 if a use that is installed would not work.
    Offline unless --online is given; it never downloads a model.
    """
    from scribeio import check_report

    # loading WhisperX takes a few seconds: show that scribe is working
    with Progress().busy("checking what scribe needs"):
        lines, code = check_report.report(config_path, Path.cwd(), online=online)
    _utf8_output(sys.stdout)
    for text in lines:
        click.echo(text)
    ctx.exit(code)


def _utf8_output(stream: Any) -> None:
    """Write UTF-8 when output goes to a file or a pipe, not a terminal.

    On Windows a redirected stdout uses the legacy code page (e.g. cp1250),
    which has no ✔ ✘ ⚠ ℹ: printing them would crash. A terminal needs
    nothing (click writes to the Windows console directly).
    """
    if not stream.isatty() and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def _interactive() -> bool:
    """True when a person can answer questions (stdin is a terminal)."""
    return sys.stdin.isatty()


def _find_config(folder: Path, progress: Progress) -> Path | None:
    """scribe.yaml in `folder`, announced when found."""
    found = init.find_config(folder)
    if found:
        progress.console.print(
            f"using config {found.name} (next to the input)", markup=False
        )
    return found


def _create_config(
    source: Path | None,
    folder: Path,
    language: str | None,
    force: bool,
    progress: Progress,
    only: init.Only = None,
    full: bool = False,
) -> None:
    """Write scribe.yaml (+ glossary.txt) from `source`; say what to do next."""
    speakers = music = None
    if source is not None:
        patterns = load_config().settings["input"]["music_patterns"]
        speakers, music = init.speakers_from(source, patterns)
    written, kept = init.write(folder, speakers, music, language, force, only, full)
    # soft_wrap: paths and the command are never broken across lines, so
    # they can be copied from the terminal.
    names = ", ".join(p.name for p in written)
    out = progress.console
    out.print(f"✔ created {names} in {folder}", markup=False, soft_wrap=True)
    for path in kept:
        out.print(f"kept existing {path.name}", markup=False, soft_wrap=True)
    target = f'"{source}"' if source is not None else "<recording>"
    # scribe finds scribe.yaml only next to the recording; elsewhere it needs -c.
    beside = source is not None and folder.resolve() == source.resolve().parent
    config = "" if beside else f' -c "{folder / init.CONFIG_NAME}"'
    edit = "the settings" if only == "master" else "the speakers (and glossary)"
    out.print(f"Next: edit {edit} in {init.CONFIG_NAME}, then run:", markup=False)
    # a .json is transcription output: mastering needs the recording, and
    # the transcript is rebuilt with merge
    is_json = source is not None and source.suffix.lower() == ".json"
    if only != "transcribe":
        recording = "<recording>" if is_json else target
        out.print(f"  scribe master {recording}{config}", markup=False, soft_wrap=True)
    if only != "master":
        command = "merge" if is_json else "transcribe"
        out.print(f"  scribe {command} {target}{config}", markup=False, soft_wrap=True)


def _glossary(cfg: Config) -> Glossary:
    return load_glossary(cfg.glossary_path) if cfg.glossary_path else Glossary()


def _paths(out_dir: Path, name: str) -> dict[str, Path]:
    """The three output files for a session name."""
    return {
        "md": out_dir / f"{name}_transcript.md",
        "json": out_dir / f"{name}_transcript.json",
        "report": out_dir / f"{name}_transcript-report.md",
    }


def _json_name(json_path: Path) -> str:
    """Session name of a .json: its stem without a trailing ``_transcript``."""
    stem = json_path.stem
    return stem.removesuffix("_transcript") or stem


def _merge_and_write(
    raw: RawSession, cfg: Config, gloss: Glossary, paths: dict[str, Path]
) -> pipeline.MergeResult:
    """Run the merge pipeline and write the transcript and the report."""
    result = pipeline.run(raw, cfg.settings["transcribe"], cfg.speakers, gloss)
    markdown.write(paths["md"], raw, result)
    report.write(paths["report"], raw, result)
    return result


def _summary(
    progress: Progress, paths: dict[str, Path], result: pipeline.MergeResult
) -> None:
    progress.console.print(
        f"{paths['md'].name}: {len(result.lines)} lines, "
        f"{len(result.dropped)} segments dropped, {len(result.bleed)} possible "
        f"bleed - see {paths['report'].name}",
        markup=False,
    )
