"""scribe — transcripts of TTRPG sessions recorded with Craig (Discord)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scribeio")
except PackageNotFoundError:  # source tree without `pip install -e .`
    __version__ = "0.0.0"
