"""MoCoTool local web server (FastAPI).

Serves the single-page UI and drives the engine through a simple one-job-at-a-time
model with polling for progress. Single-mode results keep the original + corrected
volumes in memory so the slice/echo viewer can render panels on demand.

Run via ../run_app.py (which starts uvicorn and opens the browser).
"""
from __future__ import annotations

import glob
import io
import os
import subprocess
import sys
import threading
import time
import uuid
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import paths as _paths                                # noqa: E402
_paths.ensure_importable()
_BUNDLE = _paths.bundle_dir()
_WEIGHTS_ROOT = _paths.weights_root()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from engine import load_engine                       # noqa: E402
from engine.reporting import write_summary, clean_for_json  # noqa: E402

app = FastAPI(title="MoCoTool")

# ----------------------------------------------------------------- global state
_ENGINE = None
_ENGINE_LOCK = threading.Lock()
_JOBS: Dict[str, "Job"] = {}
_ACTIVE = threading.Lock()          # one inference job at a time

INPUT_GLOB = "ima_comb_*.mat"


class Job:
    def __init__(self, mode_ui: str):
        self.id = uuid.uuid4().hex[:12]
        self.mode_ui = mode_ui          # 'single' | 'batch'
        self.status = "queued"          # queued|running|done|error|cancelled
        self.progress: List[str] = []
        self.total = 0
        self.done = 0
        self.results: List[Dict] = []
        self.error = ""
        self.cancel = False
        # single-mode viz volumes (in memory)
        self.orig: Optional[np.ndarray] = None
        self.corr: Optional[np.ndarray] = None
        self.summary_path = ""

    def log(self, msg: str):
        self.progress.append(msg)
        if len(self.progress) > 500:
            self.progress = self.progress[-500:]


def _engine():
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = load_engine(_WEIGHTS_ROOT)
    return _ENGINE


def _gate_override(opt: str, run_option: str):
    # run_option: detect | correct | detect_correct
    if run_option == "correct":
        return True                      # unconditional correction
    if opt == "force":
        return True
    if opt == "skip":
        return False
    return None


def _mode_for(run_option: str) -> str:
    return {"detect": "detect", "correct": "correct", "detect_correct": "detect_correct"}[run_option]


# --------------------------------------------------------------------- runners
def _run_single(job: Job, params: Dict):
    eng = _engine()
    mode = _mode_for(params["run_option"])
    out_dir = params.get("output_dir") or os.path.join(os.path.dirname(params["input_path"]), "mocotool_output")
    job.status = "running"; job.total = 1
    res = eng.process_volume(
        params["input_path"], mode, output_dir=out_dir,
        force_correct=_gate_override(params.get("gate", "auto"), params["run_option"]),
        threshold=params.get("threshold"),
        save_output=(mode != "detect"),
        max_iters=params.get("max_iters"),
        save_intermediate=params.get("save_intermediate", False),
        progress=job.log,
    )
    # stash volumes for the viewer, then strip from the JSON result
    if "_corrected_volume" in res:
        job.corr = res.pop("_corrected_volume")
    from engine import io_utils
    try:
        job.orig, _ = io_utils.load_volume(params["input_path"])
    except Exception:
        job.orig = None
    job.results = [clean_for_json(res)]
    job.done = 1
    job.status = "error" if res["status"] == "error" else "done"


def _run_batch(job: Job, params: Dict):
    eng = _engine()
    mode = _mode_for(params["run_option"])
    folder = params["input_path"]
    files = sorted(glob.glob(os.path.join(folder, INPUT_GLOB)))
    files = [f for f in files if "_corrected" not in f and "_fix192" not in f]
    out_dir = params.get("output_dir") or os.path.join(folder, "mocotool_output")
    job.total = len(files); job.status = "running"
    if not files:
        job.status = "error"; job.error = f"No {INPUT_GLOB} in {folder}"; return
    results = []
    for i, f in enumerate(files, 1):
        if job.cancel:
            job.status = "cancelled"; break
        job.log(f"[{i}/{len(files)}] {os.path.basename(f)}")
        res = eng.process_volume(
            f, mode, output_dir=out_dir,
            force_correct=_gate_override(params.get("gate", "auto"), params["run_option"]),
            threshold=params.get("threshold"),
            save_output=(mode != "detect"),
            max_iters=params.get("max_iters"),
            save_intermediate=params.get("save_intermediate", False),
            progress=job.log,
        )
        results.append(clean_for_json(res))
        job.done = i
        job.results = results
    os.makedirs(out_dir, exist_ok=True)
    summary = os.path.join(out_dir, "batch_summary.xlsx")
    try:
        write_summary(summary, results)
        job.summary_path = summary
    except Exception as e:
        job.log(f"summary write failed: {e}")
    if job.status == "running":
        job.status = "done"


def _run_job(job: Job, params: Dict):
    with _ACTIVE:
        try:
            (_run_single if job.mode_ui == "single" else _run_batch)(job, params)
        except Exception as e:  # pragma: no cover
            job.status = "error"; job.error = str(e); job.log(f"FATAL: {e}")


# ------------------------------------------------------------------- endpoints
@app.get("/", response_class=HTMLResponse)
def index():
    with open(_paths.resource("ui", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/status")
def status():
    gpu = []
    try:
        import tensorflow as tf
        gpu = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        pass
    return {"device": "GPU" if gpu else "CPU", "gpus": gpu, "engine_loaded": _ENGINE is not None}


@app.post("/api/browse")
async def browse(request: Request):
    body = await request.json()
    kind = "folder" if body.get("kind") == "folder" else "file"
    # Self-reinvoke so tkinter runs in its own process (works frozen and in dev).
    if _paths.is_frozen():
        cmd = [sys.executable, "--browse-dialog", kind]
    else:
        cmd = [sys.executable, os.path.join(_BUNDLE, "run_app.py"), "--browse-dialog", kind]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        # last non-empty stdout line is the chosen path
        lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
        return {"path": lines[-1].strip() if lines else ""}
    except Exception as e:
        return {"path": "", "error": str(e)}


@app.post("/api/run")
async def run(request: Request):
    body = await request.json()
    mode_ui = body.get("mode_ui", "single")
    params = {
        "run_option": body.get("run_option", "detect_correct"),
        "input_path": body.get("input_path", "").strip(),
        "output_dir": (body.get("output_dir") or "").strip() or None,
        "threshold": float(body["threshold"]) if body.get("threshold") not in (None, "") else None,
        "gate": body.get("gate", "auto"),
        "max_iters": int(body["max_iters"]) if body.get("max_iters") not in (None, "") else None,
        "save_intermediate": bool(body.get("save_intermediate", False)),
    }
    if not params["input_path"] or not os.path.exists(params["input_path"]):
        return JSONResponse({"error": "input path does not exist"}, status_code=400)
    if _ACTIVE.locked():
        return JSONResponse({"error": "a job is already running"}, status_code=409)
    job = Job(mode_ui)
    _JOBS[job.id] = job
    threading.Thread(target=_run_job, args=(job, params), daemon=True).start()
    return {"job_id": job.id}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return {
        "status": job.status, "total": job.total, "done": job.done,
        "progress": job.progress[-40:], "results": job.results,
        "error": job.error, "has_corrected": job.corr is not None,
        "summary_path": job.summary_path,
    }


@app.post("/api/cancel/{job_id}")
def cancel(job_id: str):
    job = _JOBS.get(job_id)
    if job:
        job.cancel = True
    return {"ok": True}


@app.get("/api/viz/{job_id}")
def viz(job_id: str, slice: int = 0, echo: int = 0, panel: str = "orig"):
    """Render a magnitude panel (orig|corr|diff) for a slice+echo as PNG."""
    job = _JOBS.get(job_id)
    if not job or job.orig is None:
        return JSONResponse({"error": "no volume"}, status_code=404)
    H, W, S, E = job.orig.shape
    s = max(0, min(int(slice), S - 1)); e = max(0, min(int(echo), E - 1))
    orig = np.abs(job.orig[:, :, s, e]).astype(float)
    vmax = float(np.percentile(orig, 99.5)) or 1.0
    if panel == "orig" or job.corr is None:
        img, cmax, cmap = orig, vmax, "gray"
    elif panel == "corr":
        img, cmax, cmap = np.abs(job.corr[:, :, s, e]).astype(float), vmax, "gray"
    else:  # diff x5
        img = np.abs(np.abs(job.corr[:, :, s, e]).astype(float) - orig) * 5.0
        cmax, cmap = vmax, "magma"
    png = _render_png(img, cmax, cmap)
    return Response(content=png, media_type="image/png")


@app.get("/api/viz_meta/{job_id}")
def viz_meta(job_id: str):
    job = _JOBS.get(job_id)
    if not job or job.orig is None:
        return JSONResponse({"error": "no volume"}, status_code=404)
    H, W, S, E = job.orig.shape
    return {"slices": int(S), "echoes": int(E), "has_corrected": job.corr is not None}


def _render_png(img: np.ndarray, vmax: float, cmap: str) -> bytes:
    """Render a slice panel. No rotation (matches the manuscript figures), and
    upscaled so the browser does not blur a tiny PNG. 'nearest' keeps true voxels."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    H, W = img.shape
    scale = max(2.0, 560.0 / max(H, W))           # long side ~= 560 px, crisp
    fig = Figure(figsize=(W * scale / 100.0, H * scale / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(img, cmap=cmap, vmin=0, vmax=max(vmax, 1e-12), interpolation="nearest")
    buf = io.BytesIO()
    canvas.print_png(buf)
    return buf.getvalue()
