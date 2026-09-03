"""Path resolution that works both as a normal Python app and as a frozen
PyInstaller executable.

Two roots matter:
  * bundle_dir() -- where bundled code and small assets live (ui/index.html,
                    vendor/, engine/). Frozen: sys._MEIPASS. Dev: the MoCoTool folder.
  * app_dir()    -- where the executable (or MoCoTool folder) sits. The large
                    model weights live in <app_dir>/weights so they can be updated
                    without rebuilding the .exe.
"""
from __future__ import annotations

import os
import sys


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> str:
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.abspath(os.path.dirname(__file__))


def app_dir() -> str:
    if is_frozen():
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.dirname(__file__))


def resource(*parts) -> str:
    """A bundled resource (code / index.html / vendor)."""
    return os.path.join(bundle_dir(), *parts)


def weights_root() -> str:
    """Folder that CONTAINS the 'weights/' directory (external, beside the exe)."""
    return app_dir()


def ensure_importable() -> None:
    """Make `engine`, `vendor`, `ui` importable in both modes."""
    b = bundle_dir()
    if b not in sys.path:
        sys.path.insert(0, b)
