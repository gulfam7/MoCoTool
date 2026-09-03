"""Odd in-plane matrix-size handling.

Models expect width = TARGET_WIDTH (192). Some volumes arrive wider (e.g. 254)
because of a larger phase FOV at the same voxel size -> the extra columns are
empty background. We detect the head robustly (2-D connected component on the
central brain slices) and crop a TARGET_WIDTH window centred on it. A crop
changes no intensities and never distorts anatomy.

If the head genuinely exceeds TARGET_WIDTH (a real resolution difference), we
raise instead of silently k-space-resampling, because that would change the
voxel size and squeeze the brain. This mirrors fix_odd_dimensions.m.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DimFixResult:
    changed: bool
    operation: str          # 'none' | 'crop_head_centred' | 'pad_to_192'
    orig_shape: tuple
    new_shape: tuple
    detail: str
    crop_start: int = 0     # for crop: columns kept were [crop_start, crop_end)
    crop_end: int = 0
    orig_width: int = 0


def _head_cols(vol: np.ndarray, frac: float = 0.10, lo: float = 0.35, hi: float = 0.65):
    """Median [c1, c2] head column span over central slices via largest CC."""
    from scipy import ndimage as ndi

    H, W, S, E = vol.shape
    e1 = np.abs(vol[..., 0])
    s0, s1 = int(lo * S), int(hi * S)
    c1s, c2s, hs = [], [], []
    for s in range(s0, max(s0 + 1, s1)):
        img = e1[:, :, s].astype(float)
        mx = img.max()
        if mx <= 0:
            continue
        m = ndi.binary_fill_holes(img >= frac * mx)
        lab, n = ndi.label(m)
        if n == 0:
            continue
        big = 1 + int(np.argmax(ndi.sum(np.ones_like(lab), lab, range(1, n + 1))))
        cols = np.flatnonzero((lab == big).any(axis=0))
        rows = np.flatnonzero((lab == big).any(axis=1))
        c1s.append(cols[0]); c2s.append(cols[-1]); hs.append(rows[-1] - rows[0] + 1)
    if not c1s:
        raise RuntimeError("Head detection failed (empty mask).")
    return int(np.median(c1s)), int(np.median(c2s)), int(np.median(hs))


def fix_width(vol: np.ndarray, target_width: int = 192) -> tuple[np.ndarray, DimFixResult]:
    """Return (fixed_volume, DimFixResult). Crops wider volumes, edge-pads narrow ones."""
    H, W, S, E = vol.shape
    orig = vol.shape
    if W == target_width:
        return vol, DimFixResult(False, "none", orig, orig, "already target width")

    if W < target_width:
        pad = target_width - W
        out = np.pad(vol, ((0, 0), (0, pad), (0, 0), (0, 0)), mode="edge")
        return out, DimFixResult(True, "pad_to_192", orig, out.shape,
                                 f"edge-padded width {W}->{target_width}", orig_width=W)

    # W > target_width -> crop if the head fits, else refuse.
    c1, c2, _ = _head_cols(vol)
    headW = c2 - c1 + 1
    if headW > target_width:
        raise ValueError(
            f"Head width {headW}px exceeds target {target_width}px for this volume "
            f"(shape {orig}). This looks like a real resolution difference, not extra "
            f"FOV. Verify the voxel size before processing; k-space downsampling is not "
            f"applied automatically because it changes voxel size and distorts the brain."
        )
    ctr = (c1 + c2) // 2
    start = int(np.clip(round(ctr - target_width / 2), 0, W - target_width))
    end = start + target_width
    out = vol[:, start:end, :, :]
    return out, DimFixResult(True, "crop_head_centred", orig, out.shape,
                             f"head cols[{c1}..{c2}] w={headW}px -> crop cols {start}:{end-1}",
                             crop_start=start, crop_end=end, orig_width=W)


def restore_to_original(corrected_192: np.ndarray, original_vol: np.ndarray,
                        dim: "DimFixResult") -> np.ndarray:
    """Map a corrected TARGET_WIDTH volume back to the input's native width so the
    saved output matches the input dimensions.

      pad_to_192       -> drop the padded columns (return native width).
      crop_head_centred-> re-embed the corrected window into a full-width frame,
                          keeping the original (background) columns outside the crop.
      none             -> unchanged.
    """
    if dim.operation == "pad_to_192":
        return corrected_192[:, : dim.orig_width, :, :]
    if dim.operation == "crop_head_centred":
        out = np.array(original_vol, dtype=corrected_192.dtype)  # keep original background
        out[:, dim.crop_start:dim.crop_end, :, :] = corrected_192
        return out
    return corrected_192
