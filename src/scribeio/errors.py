"""Errors scribe expects and reports to the user.

The CLI shows a `ScribeError` as one red line and exits with code 1, without
a traceback. Any other exception is a bug and keeps its traceback. Also the
error for a user's text file that is not UTF-8 (`decode_utf8`).
"""


class ScribeError(Exception):
    """A problem with the user's input, config, environment or outputs."""


class InstallError(ScribeError):
    """scribe or a tool it needs is not installed or not usable.

    The CLI adds a pointer to ``scribe check`` after it.
    """


def decode_utf8(data: bytes, label: str) -> str:
    """Decode a text file the user may have edited (a BOM is allowed).

    Args:
        data: The file's bytes.
        label: File name for the message (no path).

    Raises:
        ScribeError: If the data is not UTF-8, naming the first bad line.
    """
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        line = data[: e.start].count(b"\n") + 1
        raise ScribeError(f"{label}:{line}: not UTF-8; save it as UTF-8") from None
