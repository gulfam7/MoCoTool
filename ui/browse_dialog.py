"""Native OS file/folder picker. Exposes open_dialog() (called via the app's
self-reinvoke in both frozen and dev modes) and also works as a standalone
script. tkinter runs in its own short-lived process, never on server threads."""
from __future__ import annotations

import sys


def open_dialog(kind: str = "file") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return ""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if kind == "folder":
        path = filedialog.askdirectory(title="Select a folder of .mat files")
    else:
        path = filedialog.askopenfilename(
            title="Select a .mat file",
            filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
        )
    root.destroy()
    return path or ""


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "file"
    print(open_dialog(kind))
