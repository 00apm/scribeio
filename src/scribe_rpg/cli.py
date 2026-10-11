"""Command-line interface: `scribe transcribe` and `scribe merge`.

Commands stay thin: they read options, call the packages and report
progress. Errors scribe expects (`ScribeError`) are shown as one red line
with exit code 1; anything else is a bug and keeps its traceback.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TypeVar

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


@click.group(cls=_Group)
@click.version_option(__version__, prog_name="scribe")
def main() -> None:
    """Transcripts of TTRPG sessions recorded with Craig (Discord)."""


@main.command()
@click.argument(
    "input_path", metavar="INPUT", type=click.Path(exists=True, path_type=Path)
)
@_common
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
    force: bool,
    yes: bool,
) -> None:
    """Transcribe a Craig recording.

    INPUT is the Craig .zip, or a folder with info.txt and <n>-<account>.flac
    tracks. Writes <name>.md (transcript), <name>.json (raw data for
    `scribe merge`) and "<name> - raport.md" (quality report). Needs WhisperX
    (see README); an NVIDIA GPU is strongly recommended.
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
            rec = stack.enter_context(craig.open_recording(input_path, patterns))
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
            if t.account not in cfg.speakers:
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
    """Rebuild transcript and report from a <name>.json, without the GPU.

    Use it after changing speakers, glossary or cleanup settings. Overwrites
    <name>.md and "<name> - raport.md"; never changes the .json.
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
    paths = _paths(out_dir or json_path.resolve().parent, name or json_path.stem)
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
        "md": out_dir / f"{name}.md",
        "json": out_dir / f"{name}.json",
        "report": out_dir / f"{name} - raport.md",
    }


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
