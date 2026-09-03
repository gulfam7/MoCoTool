#!/usr/bin/env python3
"""MoCoTool entry point (also the PyInstaller .exe entry).

Normal launch:
    python run_app.py            # starts local server, opens the browser

Internal self-reinvoke (used by the app's Browse buttons; works frozen too):
    <this> --browse-dialog file|folder   # opens a native picker, prints the path, exits
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import time
import webbrowser

# Make bundled packages importable in both dev and frozen modes.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    from paths import bundle_dir, ensure_importable
    ensure_importable()
except Exception:
    pass

HOST = "127.0.0.1"
PORT = 8000


def _handle_browse_flag() -> bool:
    """If invoked as a file-dialog helper, run the dialog and exit. Returns True if handled."""
    if "--browse-dialog" in sys.argv:
        i = sys.argv.index("--browse-dialog")
        kind = sys.argv[i + 1] if i + 1 < len(sys.argv) else "file"
        try:
            from ui.browse_dialog import open_dialog
            print(open_dialog(kind))
        except Exception:
            print("")
        return True
    return False


def _open_browser():
    time.sleep(1.6)
    webbrowser.open(f"http://{HOST}:{PORT}")


def main() -> int:
    if _handle_browse_flag():
        return 0

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import uvicorn
    from ui.app import app

    print(f"\n  MoCoTool running at http://{HOST}:{PORT}")
    print("  A browser tab should open automatically. Close this window to stop.\n")
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()   # required for frozen apps that spawn processes
    raise SystemExit(main())
