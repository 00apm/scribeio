"""Errors scribe expects and reports to the user.

The CLI shows a `ScribeError` as one red line and exits with code 1, without
a traceback. Any other exception is a bug and keeps its traceback.
"""


class ScribeError(Exception):
    """A problem with the user's input, config, environment or outputs."""
