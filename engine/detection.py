"""Motion detector runner (Echo-Aware SubjectGate).

Builds the subject-gate model from the bundled detection config + weights,
runs it on a volume, and produces a subject-level motion decision using an
ADAPTIVE top-k that scales with the number of slices (k/N held constant so the
calibrated threshold stays valid). Per-slice probabilities are returned too.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

from . import preprocess as P


def _ensure_detector_imports(repo_root: str):
    """Import the detector model builder from the vendored, self-contained module."""
    from vendor.detection_models import build_subject_gate_model  # type: ignore
    return build_subject_gate_model


class Detector:
    def __init__(self, config: Dict, weights_path: str, repo_root: str):
        self.cfg = config
        self.weights_path = weights_path
        self.repo_root = repo_root
        self.cls = config.get("classification", {})
        self.norm = str(config.get("data", {}).get("norm", "echo1mean"))
        self.base_lo, self.base_hi = config.get("setting", {}).get("slice_rng", [29, 75])
        self.base_total = int(config.get("setting", {}).get("base_total_slices", 96))
        self.pad192 = bool(config.get("setting", {}).get("pad_width_to_192", True))
        self.base_k = int(self.cls.get("aggregation", {}).get("topk", 14))
        self.base_n = int(self.cls.get("aggregation", {}).get("base_n", 46))
        self.threshold = float(self.cls.get("decision_threshold", 0.15))
        self.input_rep = str(self.cls.get("input_representation", "complex20")).lower()
        self._model = None

    # -- model ----------------------------------------------------------------
    def _build(self, slice_shape: Tuple[int, int, int]):
        build_subject_gate_model = _ensure_detector_imports(self.repo_root)
        model = build_subject_gate_model(
            slice_shape=slice_shape,
            model_cfg=self.cfg.get("model", {}),
            aggregation_cfg=dict(self.cls.get("aggregation", {})),
            input_representation=self.input_rep,
        )
        model.load_weights(self.weights_path)
        return model

    # -- inference ------------------------------------------------------------
    def _prep(self, vol_hwse: np.ndarray) -> Tuple[np.ndarray, int]:
        """Return (x [Nbag,H,W,2E], n_bag). Applies scaled slice range + pad + echo1mean."""
        nxye = P.swap_to_nxye(vol_hwse)                    # [S,H,W,E]
        nxye, _, _ = P.normalize(nxye, self.norm)
        if self.pad192:
            nxye = P.pad_width_to(nxye, 192)
        n_slices = nxye.shape[0]
        lo, hi = P.scaled_slice_range(n_slices, self.base_lo, self.base_hi, self.base_total)
        bag = nxye[lo:hi]
        if self.input_rep in ("complex", "complex20"):
            x = P.to_complex20(bag)
        else:
            x = np.abs(bag).astype(np.float32)             # magnitude fallback
        return x, x.shape[0]

    def predict(self, vol_hwse: np.ndarray) -> Dict:
        x, n_bag = self._prep(vol_hwse)
        if self._model is None:
            self._model = self._build(tuple(int(v) for v in x.shape[1:]))
        mask = np.ones((1, n_bag), dtype=np.float32)
        outputs = self._model([x[np.newaxis, ...], mask], training=False)
        # outputs: [p_subject, p_slice, subject_logit, slice_logits]
        slice_logits = np.asarray(outputs[3]).reshape(-1)[:n_bag].astype(np.float32)
        p_slice = np.asarray(outputs[1]).reshape(-1)[:n_bag].astype(np.float32)

        k = P.adaptive_topk(n_bag, self.base_k, self.base_n)
        top = np.argsort(slice_logits)[-k:]
        p_subject = float(1.0 / (1.0 + np.exp(-float(np.mean(slice_logits[top])))))
        decision = "MOTION" if p_subject >= self.threshold else "MOTION_FREE"
        return {
            "p_subject": p_subject,
            "decision": decision,
            "threshold": self.threshold,
            "topk": int(k),
            "n_bag": int(n_bag),
            "p_slice": p_slice,
            "slice_logits": slice_logits,
        }
