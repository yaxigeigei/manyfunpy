#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IO utility functions
"""
import pickle
from pathlib import Path
from typing import Any


def load_pickle(path: str | Path) -> Any:
    # Load serialized object.
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    # Save serialized object.
    with open(path, "wb") as f:
        pickle.dump(obj, f)
