"""Command-line interface: `scribe transcribe` and `scribe merge`.

Commands stay thin: they read options, call the packages and report
progress. Errors scribe expects (`ScribeError`) are shown as one red line
with exit code 1; anything else is a bug and keeps its traceback.
"""

from __future__ import annotations

import re
import shutil
import sys
from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import click

from scribe_rpg import __version__, craig, init
from scribe_rpg.config import Config, load_config
from scribe_rpg.errors import ScribeError
from scribe_rpg.progress import Progress
from scribe_rpg.transcribe import (
    markdown,
    models,
    pipeline,
    report,
    session_json,
    whisper,
)
from scribe_rpg.transcribe.glossary import Glossary, load_glossary
from scribe_rpg.transcribe.session import RawSession, Segment, TrackInfo

if TYPE_CHECKING:
    # numpy at runtime only inside `master` / `transcribe`, never for `merge`
    from scribe_rpg.master.config import MasterConfig
    from scribe_rpg.master.excerpt import Excerpt
    from scribe_rpg.master.render import Measured

F = TypeVar("F", bound=Callable[..., Any])


class _Group(click.Group):
    """Click group that reports ScribeError as one red line, exit code 1."""

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except ScribeError as e:
            click.secho(str(e), fg="red", err=True)
            ctx.exit(1)


def _common(f: F) -> F:
    """Options shared by `transcribe` and `merge`."""
    options = [
        click.option(
            "-c", "--config", "config_path",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="Campaign config (YAML): speakers, glossary, settings.",
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
    unless --name gives one. Needs WhisperX (see README); an NVIDIA GPU is
    strongly recommended.
    """
    # Imported here: numpy is only needed for transcription, not for merge.
    from scribe_rpg.transcribe import energy

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

    # Cheap checks first: fail in seconds, before unzipping or loading models.
    with progress.step("check ffmpeg, WhisperX, CUDA"):
        warnings = whisper.preflight(wcfg)
        used_versions = whisper.versions()
    for w in warnings:
        progress.warn(w)
    with progress.step("check models"):
        needed = models.needed(wcfg)
    wcfg = _ensure_models(needed, wcfg, progress, yes)

    # The ExitStack keeps the zip's temp folder alive until outputs are written
    # and removes it afterwards, also on errors and Ctrl+C.
    with ExitStack() as stack:
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
        if transcriber.prompt_cut:
            cut = ", ".join(transcriber.prompt_cut)
            progress.warn(f"glossary names left out of the prompt: {cut}")

        cleanup = cfg.settings["transcribe"]["cleanup"]
        segments: list[Segment] = []
        duration = 0.0
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
        )
        with progress.step(f"write {paths['md'].name}"):
            paths["md"].parent.mkdir(parents=True, exist_ok=True)
            session_json.write(paths["json"], raw)
            result = _merge_and_write(raw, cfg, gloss, paths)
    _summary(progress, paths, result)


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
    help="Campaign config (YAML): speakers, master settings.",
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
    "--format", "fmt", type=click.Choice(["mp3", "m4a", "opus"]),
    help="Output format (default: master.format in the config, else mp3).",
)  # fmt: skip
@click.option(
    "--tracks", "write_tracks", is_flag=True,
    help="Also write each processed track to <name>_tracks/ (FLAC).",
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
    fmt: str | None,
    write_tracks: bool,
) -> None:
    """Mix and master a Craig recording into one audio file.

    INPUT is the Craig .zip, or a folder of FLAC tracks (one per speaker;
    without info.txt it needs --start). Writes <name>_master.<format>: every
    track through its chain, mixed, the mix chain, encoded. With --preview,
    writes only <name>_preview/ for a short excerpt: tracks/ (the excerpt of
    every track, unprocessed) and listening files at equal loudness.
    """
    used = [opt for opt, v in [("--at", at), ("--length", length)] if v is not None]
    used += ["--montage"] if montage else []
    if used and not preview:
        raise click.UsageError(f"{', '.join(used)} needs --preview")
    if at is not None and montage:
        raise click.UsageError("--at and --montage cannot be used together")
    if write_tracks and preview:
        raise click.UsageError("--tracks cannot be used with --preview")

    from scribe_rpg import ffmpeg
    from scribe_rpg.master import mix
    from scribe_rpg.master.config import resolve, unknown_players

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
    ffmpeg.require()
    if old := ffmpeg.version_warning():
        progress.warn(old)
    with progress.step("config check"):
        mix.check(mcfg)
    with ExitStack() as stack:
        with progress.step(f"open {input_path.name}"):
            patterns = cfg.settings["input"]["music_patterns"]
            rec = stack.enter_context(craig.open_recording(input_path, patterns, start))
        for t in rec.players:
            if craig.sanitize(t.account) not in cfg.speakers:
                progress.warn(
                    f"unknown account {t.account}: no speakers: entry, voice chain used"
                )
        for key in unknown_players(mcfg, [t.account for t in rec.players]):
            progress.warn(f"players: {key}: not in this recording, ignored")
        name = name or rec.start.strftime("%Y-%m-%d")
        folder = out_dir or input_path.resolve().parent
        if preview:
            clean = cfg.settings["transcribe"]["cleanup"]
            ex = _find_excerpt(rec, clean, length or 90, at, montage, progress)
            _write_preview(rec, ex, mcfg, folder / f"{name}_preview", progress)
        else:
            tracks_dir = folder / f"{name}_tracks" if write_tracks else None
            _write_master(rec, mcfg, folder / f"{name}_master.{mcfg.format}",
                          tracks_dir, force, progress)  # fmt: skip


def _find_excerpt(
    rec: craig.Recording,
    clean: dict[str, Any],
    length: int,
    at: int | None,
    montage: bool,
    progress: Progress,
) -> Excerpt:
    """Measure every track's activity and choose the preview excerpt."""
    from scribe_rpg.master import excerpt

    with progress.step("find excerpt"):
        acts = excerpt.activities(
            [t.path for t in rec.tracks], clean["frame_ms"], clean["energy_db"]
        )
        pairs = list(zip(rec.tracks, acts, strict=True))
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
    progress: Progress,
) -> None:
    """Cut the excerpt out of every track, then write the listening files."""
    from scribe_rpg.master import preview

    with progress.step(f"cut {len(rec.tracks)} tracks"):
        preview.fresh_folder(folder)
        paths = preview.cut(rec.tracks, ex.ranges, folder / "tracks")
    cut = [
        craig.Track(t.account, t.kind, p)
        for t, p in zip(rec.tracks, paths, strict=True)
    ]
    preview.write_files(cut, mcfg, folder, progress)
    progress.console.print(f"preview: {folder}", markup=False, soft_wrap=True)


def _write_master(
    rec: craig.Recording,
    mcfg: MasterConfig,
    out: Path,
    tracks_dir: Path | None,
    force: bool,
    progress: Progress,
) -> None:
    """Measure the loudness steps, then render the whole recording to `out`.

    With `tracks_dir`, the same run writes each processed track there, plus
    the recording's info.txt, so the folder is a valid input itself.
    """
    from scribe_rpg.master import mix

    for path in [out, tracks_dir]:
        if path is not None and path.exists() and not force:
            raise ScribeError(
                f"output already exists: {path.name} (use --force to overwrite)"
            )
    plans = [
        mix.TrackPlan(t, mcfg.chain_for(t), stereo=mcfg.keeps_stereo(t))
        for t in rec.tracks
    ]
    with progress.step(f"measure {len(plans)} tracks"):
        mix_gains = mix.measure_all(
            plans, mcfg.mix, mcfg.channels, _measured_printer(progress)
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    if tracks_dir is not None:
        shutil.rmtree(tracks_dir, ignore_errors=True)  # --force: start empty
        tracks_dir.mkdir()
        if rec.info_txt is not None:
            (tracks_dir / "info.txt").write_text(rec.info_txt, encoding="utf-8")
    with progress.tracks(1) as bar:
        bar.start(out.name)
        bar.set_phase("rendering")
        mix.render(plans, mcfg.mix, mix_gains, mcfg.channels, out, mcfg.format,
                   mcfg.bitrate, bar.update, tracks_dir=tracks_dir)  # fmt: skip
        bar.done(f"render {out.name}")
    progress.console.print(f"master: {out}", markup=False, soft_wrap=True)
    if tracks_dir is not None:
        progress.console.print(f"tracks: {tracks_dir}", markup=False, soft_wrap=True)


def _measured_printer(progress: Progress) -> Callable[[str, list[Measured]], None]:
    """Print each track's loudness steps: level and gain; warn on silence."""

    def show(account: str, measured: list[Measured]) -> None:
        parts = []
        for m in measured:
            if m.loudness is None or m.gain is None:
                progress.warn(f"{account}: silent, {m.step} applies no gain")
            else:
                parts.append(f"{m.step} {m.loudness:.1f} LUFS → {m.gain:+.1f} dB")
        if parts:
            progress.console.print(f"{account}: {', '.join(parts)}", markup=False)

    return show


def _show_excerpt(ex: Excerpt, forced: bool, progress: Progress) -> None:
    """Print the chosen excerpt, the montage notice and missing voices."""
    from scribe_rpg.master.excerpt import hms

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
@click.option("--language", help="Transcription language, e.g. pl (default: en).")
@click.option("--force", is_flag=True, help="Overwrite an existing scribe.yaml.")
def init_command(
    source: Path | None, out_dir: Path | None, language: str | None, force: bool
) -> None:  # fmt: skip
    """Create scribe.yaml and glossary.txt to fill in.

    With RECORDING (a Craig .zip, a track folder or a <name>.json), the
    speakers section lists its Discord accounts, ready for their names.
    scribe transcribe and scribe merge use scribe.yaml by themselves when it
    is next to the recording or the .json.
    """
    progress = Progress()
    if out_dir is None:
        out_dir = source.resolve().parent if source else Path.cwd()
    _create_config(source, out_dir, language, force, progress)


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
) -> None:
    """Write scribe.yaml (+ glossary.txt) from `source`; say what to do next."""
    speakers = music = None
    if source is not None:
        patterns = load_config().settings["input"]["music_patterns"]
        speakers, music = init.speakers_from(source, patterns)
    written, kept = init.write(folder, speakers, music, language, force)
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
    if not beside:
        target += f' -c "{folder / init.CONFIG_NAME}"'
    out.print(
        f"Next: edit the speakers (and glossary) in {init.CONFIG_NAME}, then run:",
        markup=False,
    )
    out.print(f"  scribe transcribe {target}", markup=False, soft_wrap=True)


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
