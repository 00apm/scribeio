"""Shared pytest setup: skip ffmpeg-marked tests when ffmpeg is missing."""

import shutil

import pytest


def pytest_collection_modifyitems(config, items):
    if shutil.which("ffmpeg"):
        return
    skip = pytest.mark.skip(reason="ffmpeg not in PATH")
    for item in items:
        if "ffmpeg" in item.keywords:
            item.add_marker(skip)
