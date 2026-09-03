# Building the MoCoTool executable

Produces a portable Windows folder (`dist/MoCoTool/`) that runs without a Python
install. The model **weights are kept external** (beside the exe) so they can be
swapped without rebuilding.

## Prerequisites

```
pip install -r requirements.txt
pip install pyinstaller
```

Build on the OS you will distribute to (Windows build -> Windows `.exe`).

## Build

```
python build_exe.py
```

This runs PyInstaller with `MoCoTool.spec`, then copies `weights/` next to the
exe and writes a short `READ_ME_FIRST.txt`. Output:

```
dist/MoCoTool/
  MoCoTool.exe          <- double-click to launch
  _internal/            <- bundled Python + TensorFlow + app code + ui/index.html
  weights/
    detection/  { config.json, latest_weights.h5 }
    correction/ { config.json, latest.h5 }
  READ_ME_FIRST.txt
```

Distribute the **entire** `dist/MoCoTool/` folder. Expect ~1.5-2 GB (TensorFlow).
The build takes roughly 10-20 minutes.

## Run

Double-click `MoCoTool.exe`. A console window opens and a browser tab loads the
interface at `http://127.0.0.1:8000`. Closing the console stops the server.

## Updating weights without rebuilding

Replace the `.h5` files under `dist/MoCoTool/weights/detection` or
`.../correction`. The loader picks up any single `.h5`/`.keras` in each folder
(preferring `weights.h5`, then `latest_weights.h5`/`latest.h5`, ...). Keep the
matching `config.json` if the architecture changed.

## Design notes / gotchas handled

- **Model builders are vendored** (`vendor/correction_models.py`,
  `vendor/detection_models.py`) so nothing is imported from the repo at runtime
  and PyInstaller bundles them via normal imports.
- **torch/jax excluded.** They are installed in the dev environment (from other
  work) but unused here; the spec excludes them so the build stays lean.
- **Native file dialog under a frozen exe** works via a self-reinvoke:
  `MoCoTool.exe --browse-dialog file|folder` opens a tkinter picker in its own
  process. `run_app.py` intercepts that flag before any heavy imports.
- **`tensorflow.__internal__ ... not found`** lines during the build are benign
  (virtual modules listed by `collect_all`); the build still succeeds.
- **Weights are NOT bundled** into the exe (they are large and change often);
  they live in the external `weights/` folder.

## Troubleshooting

- *Port 8000 in use*: another MoCoTool/server instance is running. Close it (or
  change `PORT` in `run_app.py`).
- *Model fails to load in the exe*: confirm `dist/MoCoTool/weights/...` contains
  both `config.json` and a `.h5`, matching the trained architecture.
- *Rebuild from scratch*: delete `build/` and `dist/`, then `python build_exe.py`.
