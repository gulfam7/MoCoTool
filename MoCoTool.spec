# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for MoCoTool (onedir).

Bundles the app + TensorFlow + all deps. Model WEIGHTS are intentionally NOT
bundled -- the build script copies the `weights/` folder next to the exe so it
can be updated without rebuilding. Build with:

    pyinstaller MoCoTool.spec --noconfirm
    (or:  python build_exe.py)
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []


def _grab(pkg):
    try:
        d, b, h = collect_all(pkg)
        datas.extend(d); binaries.extend(b); hiddenimports.extend(h)
        return True
    except Exception as e:
        print(f"[spec] collect_all({pkg!r}) skipped: {e}")
        return False


# --- heavy / dynamically-loaded packages ------------------------------------
_grab("tensorflow")
_grab("tensorflow_intel")     # Windows pip distribution name; harmless if absent
_grab("keras")
_grab("scipy")
_grab("h5py")
_grab("matplotlib")
_grab("openpyxl")
_grab("pandas")
_grab("google.protobuf")

# --- web server submodules (uvicorn loads these dynamically) -----------------
for pkg in ("uvicorn", "fastapi", "starlette", "anyio"):
    try:
        hiddenimports.extend(collect_submodules(pkg))
    except Exception as e:
        print(f"[spec] collect_submodules({pkg!r}) skipped: {e}")

# --- our own lazily-imported modules ----------------------------------------
hiddenimports += [
    "vendor", "vendor.correction_models", "vendor.detection_models",
    "engine", "engine.pipeline", "engine.detection", "engine.correction",
    "engine.dimensions", "engine.preprocess", "engine.io_utils", "engine.reporting",
    "ui", "ui.app", "ui.browse_dialog", "paths",
    "scipy.io", "scipy.ndimage",
]

# --- bundled data assets (code-adjacent) ------------------------------------
datas += [
    ("ui/index.html", "ui"),
]


a = Analysis(
    ["run_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Heavy packages that are installed in the dev env but NOT used by MoCoTool
        # (the engine is TensorFlow-only). Excluding them keeps the build lean.
        "torch", "torchvision", "torchaudio", "torchgen", "functorch",
        "jax", "jaxlib", "tensorflow_probability",
        "tkinter.test", "test", "tests", "notebook", "IPython", "ipykernel",
        "cupy", "sphinx",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MoCoTool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,          # keep the console: shows the URL and closing it stops the server
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MoCoTool",
)
