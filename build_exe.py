#!/usr/bin/env python3
"""Build the MoCoTool .exe (onedir) and stage the external weights folder.

    python build_exe.py

Produces:  dist/MoCoTool/MoCoTool.exe   (+ _internal/, + weights/ copied beside it)

Notes
-----
* TensorFlow makes the build large (multi-GB) and slow (10-20 min). This is normal.
* Model weights are NOT bundled into the exe; they are copied to
  dist/MoCoTool/weights/ so they can be swapped without rebuilding.
* Run this on the same OS you will distribute to (Windows -> Windows .exe).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
DIST = os.path.join(ROOT, "dist", "MoCoTool")


def _check_pyinstaller():
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed. Install it with:\n    pip install pyinstaller")
        sys.exit(1)


def _build():
    print("=== Running PyInstaller (this takes several minutes for TensorFlow) ===")
    cmd = [sys.executable, "-m", "PyInstaller", "MoCoTool.spec", "--noconfirm", "--clean"]
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0:
        print(f"PyInstaller failed (exit {rc}).")
        sys.exit(rc)


def _stage_weights():
    src = os.path.join(ROOT, "weights")
    dst = os.path.join(DIST, "weights")
    if not os.path.isdir(src):
        print(f"WARNING: {src} not found; no weights staged.")
        return
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    print(f"=== Copying weights -> {dst} ===")
    shutil.copytree(src, dst)


def _write_readme():
    txt = (
        "MoCoTool — portable build\n"
        "=========================\n\n"
        "Double-click MoCoTool.exe. A console window opens and a browser tab\n"
        "loads the interface at http://127.0.0.1:8000. Close the console to stop.\n\n"
        "The 'weights/' folder next to the exe holds the trained models:\n"
        "  weights/detection/  { config.json, *.h5 }\n"
        "  weights/correction/ { config.json, *.h5 }\n"
        "You can replace the .h5 files with newer weights without rebuilding.\n"
    )
    with open(os.path.join(DIST, "READ_ME_FIRST.txt"), "w", encoding="utf-8") as f:
        f.write(txt)


def main():
    _check_pyinstaller()
    _build()
    if not os.path.isdir(DIST):
        print(f"Expected output folder not found: {DIST}")
        sys.exit(1)
    _stage_weights()
    _write_readme()
    exe = os.path.join(DIST, "MoCoTool.exe")
    print("\n=== Build complete ===")
    print(f"  App:     {exe}")
    print(f"  Weights: {os.path.join(DIST, 'weights')}")
    print("  Distribute the entire 'dist/MoCoTool' folder.")


if __name__ == "__main__":
    main()
