"""MoCoTool orchestration engine.

One entry point -- process_volume(path, mode, ...) -- that performs:
    load -> dimension-fix -> (detect / correct / detect+correct loop) -> save -> result dict

Modes:
    'detect'          : detection only. No output volume saved.
    'correct'         : correction only (unconditional, full volume).
    'detect_correct'  : gated iterative loop. Detect; if MOTION, correct; re-detect;
                        repeat up to max_iters (default 3) or until MOTION_FREE.

The result dict feeds both the CLI and the batch Excel writer. Denormalization is
always applied before saving (native amplitude scale).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import io_utils, preprocess as P
from .correction import Corrector
from .detection import Detector
from .dimensions import fix_width, restore_to_original


@dataclass
class EngineConfig:
    repo_root: str
    detection_config: Dict
    detection_weights: str
    correction_config: Dict
    correction_weights: str
    target_width: int = 192
    max_iters: int = 3


class Engine:
    """Holds the two model runners (built lazily, reused across a batch)."""

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.detector = Detector(cfg.detection_config, cfg.detection_weights, cfg.repo_root)
        self.corrector = Corrector(cfg.correction_config, cfg.correction_weights, cfg.repo_root)

    # ------------------------------------------------------------------ public
    def process_volume(
        self,
        input_path: str,
        mode: str,
        output_dir: Optional[str] = None,
        force_correct: Optional[bool] = None,   # override the gate: True=always, False=never
        threshold: Optional[float] = None,       # override detection threshold
        save_output: bool = True,
        max_iters: Optional[int] = None,         # override loop budget for this call
        save_intermediate: bool = False,         # also save each iteration's output
        progress: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        t0 = time.time()
        log = progress or (lambda m: None)
        subject = io_utils.subject_id_from_path(input_path)
        result: Dict = {
            "subject": subject, "input_file": input_path, "mode": mode,
            "status": "ok", "error": "",
        }
        try:
            if threshold is not None:
                self.detector.threshold = float(threshold)

            original_vol, varname = io_utils.load_volume(input_path)  # [H,W,S,E] complex, native dims
            result["orig_shape"] = list(original_vol.shape)

            # 1) dimension fix (feed the model at target width; output restored to native dims later)
            vol, dim = fix_width(original_vol, self.cfg.target_width)
            result["dim_fixed"] = dim.changed
            result["dim_op"] = dim.operation
            result["dim_detail"] = dim.detail
            result["shape"] = list(vol.shape)

            # 2) dispatch by mode
            if mode == "detect":
                self._run_detect(vol, result, log)
            elif mode == "correct":
                self._run_correct(vol, original_vol, dim, varname, result, output_dir, force_correct, save_output, log)
            elif mode == "detect_correct":
                budget = int(max_iters) if max_iters else self.cfg.max_iters
                self._run_loop(vol, original_vol, dim, varname, result, output_dir,
                               force_correct, save_output, budget, save_intermediate, log)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        except Exception as exc:  # keep the batch alive
            result["status"] = "error"
            result["error"] = str(exc)
            log(f"ERROR {subject}: {exc}")

        result["seconds"] = round(time.time() - t0, 2)
        return result

    # ----------------------------------------------------------------- private
    def _run_detect(self, vol, result, log):
        log("detecting...")
        d = self.detector.predict(vol)
        result.update({
            "p_subject": d["p_subject"], "decision": d["decision"],
            "threshold": d["threshold"], "topk": d["topk"], "n_bag": d["n_bag"],
            "corrected": False, "iterations": 0,
            "p_slice": d["p_slice"].tolist(),
        })

    def _run_correct(self, vol, original_vol, dim, varname, result, output_dir, force_correct, save_output, log):
        # unconditional correction unless force_correct is explicitly False
        if force_correct is False:
            result.update({"corrected": False, "iterations": 0, "decision": "skipped"})
            return
        log("CORRECTING: performing motion correction (full volume, this can take a few minutes)...")
        corrected = self.corrector.predict(vol)
        corrected = restore_to_original(corrected, original_vol, dim)   # match input dims
        result.update({"corrected": True, "iterations": 1})
        if save_output:
            result["output_file"] = self._save(corrected, varname, result["subject"], output_dir, log)
        result["_corrected_volume"] = corrected  # in-memory for single-mode viz (dropped before Excel)

    def _run_loop(self, vol, original_vol, dim, varname, result, output_dir,
                  force_correct, save_output, budget, save_intermediate, log):
        history: List[Dict] = []
        current = vol
        iters = 0
        corrected_any = False
        final_corrected = None

        for i in range(int(budget)):
            d = self.detector.predict(current)
            history.append({"iter": i + 1, "p_subject": d["p_subject"], "decision": d["decision"]})
            log(f"iter {i+1}/{budget}: detected p={d['p_subject']:.3f} -> {d['decision']}")

            do_correct = (d["decision"] == "MOTION")
            if force_correct is True:
                do_correct = True
            if force_correct is False:
                do_correct = False

            if not do_correct:
                break
            # progress marker the UI turns into an animated "Performing Motion Correction"
            log(f"CORRECTING iter {i+1}/{budget}: performing motion correction (full volume, this can take a few minutes)...")
            current = self.corrector.predict(current)
            corrected_any = True
            final_corrected = current
            iters += 1
            if save_intermediate and save_output:
                inter = restore_to_original(current, original_vol, dim)
                p = self._save(inter, varname, f"{result['subject']}_iter{i+1}", output_dir, log)
                history[-1]["output_file"] = p

        # capture final decision after last (re-)detection
        final = self.detector.predict(current)
        result.update({
            "p_subject_initial": history[0]["p_subject"],
            "p_subject_final": final["p_subject"],
            "decision": final["decision"],
            "threshold": final["threshold"],
            "topk": final["topk"], "n_bag": final["n_bag"],
            "corrected": corrected_any,
            "iterations": iters,
            "loop_history": history,
            "p_slice": final["p_slice"].tolist(),
        })
        if corrected_any:
            final_corrected = restore_to_original(final_corrected, original_vol, dim)  # match input dims
            result["_corrected_volume"] = final_corrected
            if save_output:
                result["output_file"] = self._save(final_corrected, varname, result["subject"], output_dir, log)

    def _save(self, corrected_hwse, varname, subject, output_dir, log) -> str:
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, f"ima_comb_{subject}_corrected.mat")
        io_utils.save_volume(out, corrected_hwse, varname,
                             extra={"mat_output_scale": np.array("input_native")})
        log(f"saved {out}")
        return out


def _find_weights(folder: str) -> str:
    """Locate the .h5/.keras weights file in a weights subfolder (name-agnostic)."""
    import glob
    prefer = ["weights.h5", "latest_weights.h5", "latest.h5", "best_weights.h5",
              "best.h5", "final_weights.h5", "final.h5"]
    for name in prefer:
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            return p
    cand = sorted(glob.glob(os.path.join(folder, "*.h5")) + glob.glob(os.path.join(folder, "*.keras")))
    if len(cand) == 1:
        return cand[0]
    if not cand:
        raise FileNotFoundError(f"No .h5/.keras weights found in {folder}")
    raise FileNotFoundError(f"Multiple weight files in {folder}; keep one: {[os.path.basename(c) for c in cand]}")


# ------------------------------------------------------------------- config load
def load_engine(weights_root: str, repo_root: str = "", max_iters: int = 3) -> Engine:
    """Build an Engine from the weights/ folder layout:
        <weights_root>/weights/detection/{config.json, *.h5}
        <weights_root>/weights/correction/{config.json, *.h5}
    `repo_root` is accepted for backward compatibility but unused (model builders
    are vendored)."""
    def _load(sub):
        d = os.path.join(weights_root, "weights", sub)
        with open(os.path.join(d, "config.json")) as f:
            cfg = json.load(f)
        w = _find_weights(d)
        return cfg, w

    det_cfg, det_w = _load("detection")
    cor_cfg, cor_w = _load("correction")
    return Engine(EngineConfig(
        repo_root=repo_root,
        detection_config=det_cfg, detection_weights=det_w,
        correction_config=cor_cfg, correction_weights=cor_w,
        max_iters=max_iters,
    ))
