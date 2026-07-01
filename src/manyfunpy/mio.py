#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IO utility functions
"""
import gzip
import pickle
from pathlib import Path, WindowsPath
from typing import Any


class WindowsPathUnpickler(pickle.Unpickler):
    # Load POSIX-authored path objects on Windows.
    def find_class(self, module, name):
        if (module, name) == ("pathlib", "PosixPath"):
            return WindowsPath
        return super().find_class(module, name)


def load_pickle(path: str | Path) -> Any:
    """Load a pickle file, using gzip decompression when the path ends in `.gz`."""
    # Load serialized object.
    path = Path(path)
    with _open_pickle(path, "rb") as f:
        return WindowsPathUnpickler(f).load()


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save a pickle file, using gzip compression when the path ends in `.gz`."""
    # Save serialized object.
    path = Path(path)
    with _open_pickle(path, "wb") as f:
        pickle.dump(obj, f)


def _open_pickle(path: Path, mode: str):
    """Open a raw or gzip-compressed pickle stream based on the file suffix."""
    # Select compression from file extension.
    if path.suffix == ".gz":
        return gzip.open(path, mode)
    return open(path, mode)
