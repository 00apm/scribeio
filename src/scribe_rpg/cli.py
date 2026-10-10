"""Command-line interface: `scribe transcribe`, `scribe merge`."""

from __future__ import annotations

import click

from scribe_rpg import __version__


@click.group()
@click.version_option(__version__, prog_name="scribe")
def main() -> None:
    """Transcripts of TTRPG sessions recorded with Craig (Discord)."""
