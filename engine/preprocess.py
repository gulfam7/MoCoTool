"""Shared preprocessing: axis swaps, normalization (+ inverse), channel packing,
slice cropping. Mirrors the conventions in correction_model/preprocess.py and
EchoAwareSubjectGate so the tool feeds each model exactly what it was trained on.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---- axis conventions --------------------------------------------------------
def swap_to_nxye(volume_hwse: np.ndarray) -> np.ndarray:
    """[H,W,S,E] -> [S,H,W,E]."""
    return np.swapaxes(np.swapaxes(volume_hwse, 0, 2), 1, 2)


def swap_to_hwse(nxye: np.ndarray) -> np.ndarray:
    """[S,H,W,E] -> [H,W,S,E]."""
    return np.swapaxes(np.swapaxes(nxye, 1, 2), 0, 2)


# ---- normalization -----------------------------------------------------------
def norm_params(nxye: np.ndarray, method: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (offset, scale) so that x_norm = (x - offset) / (scale + eps)."""
    m = (method or "none").lower()
    if m in ("none", "null"):
        return np.asarray(0.0), np.asarray(1.0, dtype=np.float32)
    if m == "midcube":
        s = nxye.shape[0]
        return np.asarray(0.0), np.asarray(np.mean(np.abs(nxye[s // 2, ...])), dtype=np.float32)
    if m == "echo1mean":
        mag = np.abs(nxye[..., 0])
        nz = mag > 0
        ref = np.mean(mag[nz]) if np.any(nz) else np.mean(mag)
        return np.asarray(0.0), np.asarray(ref, dtype=np.float32)
    if m == "max":
        return np.asarray(0.0), np.amax(np.abs(nxye), axis=(1, 2), keepdims=True)
    if m == "mean":
        return np.asarray(0.0), np.mean(np.abs(nxye), axis=(1, 2), keepdims=True)
    raise ValueError(f"Unsupported norm method: {method}")


def normalize(nxye: np.ndarray, method: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    off, scale = norm_params(nxye, method)
    return (nxye - off) / (scale + 1e-12), off, scale


def denormalize(nxye: np.ndarray, offset, scale) -> np.ndarray:
    return nxye * (scale + 1e-12) + offset


# ---- width padding (matches pad_width_190_to_192) ----------------------------
def pad_width_to(nxye: np.ndarray, target_w: int = 192) -> np.ndarray:
    w = nxye.shape[2]
    if w == target_w:
        return nxye
    if w < target_w:
        cfg = [(0, 0)] * nxye.ndim
        cfg[2] = (0, target_w - w)
        return np.pad(nxye, cfg, mode="edge")
    return nxye[:, :, :target_w, ...]


# ---- channel packing ---------------------------------------------------------
def to_complex20(nxye: np.ndarray) -> np.ndarray:
    """[S,H,W,E] complex -> [S,H,W,2E] real, layout [Re1..ReE, Im1..ImE] (detector)."""
    return np.concatenate((nxye.real, nxye.imag), axis=-1).astype(np.float32)


def to_complex_lastdim(nxye: np.ndarray) -> np.ndarray:
    """[S,H,W,E] complex -> [S,H,W,E,2] real (corrector)."""
    return np.concatenate((nxye.real[..., None], nxye.imag[..., None]), axis=-1).astype(np.float32)


def from_complex_lastdim(pred: np.ndarray) -> np.ndarray:
    """[S,H,W,E,2] real -> [S,H,W,E] complex."""
    return (pred[..., 0] + 1j * pred[..., 1]).astype(np.complex64)


# ---- slice range -------------------------------------------------------------
def scaled_slice_range(n_slices: int, base_lo: int = 29, base_hi: int = 75,
                       base_total: int = 96) -> Tuple[int, int]:
    """Scale the detection slice range proportionally to the volume slice count.
    [29,75] on 96 slices -> same *fractions* on any slice count."""
    lo = int(round(base_lo / base_total * n_slices))
    hi = int(round(base_hi / base_total * n_slices))
    lo = max(0, min(lo, n_slices - 1))
    hi = max(lo + 1, min(hi, n_slices))
    return lo, hi


def adaptive_topk(n_bag: int, base_k: int = 14, base_n: int = 46) -> int:
    """Keep k/N constant so the calibrated threshold stays valid.
    k=14 at N=46 ; doubles when N doubles."""
    k = int(round(base_k / base_n * n_bag))
    return max(1, min(k, n_bag))
