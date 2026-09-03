"""Motion corrector runner (ME-PARNet / attention_phys).

Builds the corrector from the bundled correction config + weights, runs it on
the FULL volume (all slices; no slice-range crop), and denormalizes the output
back to the input's native amplitude scale before returning it. Denormalization
is essential -- saving in the normalized scale is what caused the earlier
downstream R2* amplitude discrepancy.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np

from . import preprocess as P

ATTN_CORR_VARIANTS = {"attn_corr", "attn_corr_phys"}


def _ensure_corrector_imports(repo_root: str):
    """Import the corrector model builder from the vendored, self-contained module."""
    from vendor.correction_models import build_unet3d_img_variant  # type: ignore
    build_unet3d_attn_corr_variant = None  # attn_corr variants not used by the default model
    return build_unet3d_img_variant, build_unet3d_attn_corr_variant


class Corrector:
    def __init__(self, config: Dict, weights_path: str, repo_root: str):
        self.cfg = config
        self.weights_path = weights_path
        self.repo_root = repo_root
        self.train = config.get("train", {})
        self.variant = str(self.train.get("variant", "attention_phys"))
        self.norm = str(config.get("data", {}).get("norm", "midcube"))
        self.pad192 = bool(config.get("setting", {}).get("pad_width_to_192", True))
        self.batch_size = int(config.get("train", {}).get("infer_batch_size", 4))
        self._model = None

    def _build(self, sample_shape: Tuple[int, ...]):
        build_img, build_attn = _ensure_corrector_imports(self.repo_root)
        p = self.cfg["method"]["unet3d"]
        common = dict(
            input_shape=sample_shape,
            variant=self.variant,
            output_channel=int(p.get("output_channel", 2)),
            filters_root=int(p.get("filters_root", 32)),
            conv_times=int(p.get("conv_times", 3)),
            up_down_times=int(p.get("up_down_times", 4)),
            num_echoes=int(sample_shape[2]),
            attn_heads=int(self.train.get("attn_heads", 2)),
            attn_key_dim=int(self.train.get("attn_key_dim", 16)),
            if_relu=bool(p.get("if_relu", False)),
            if_residule=bool(p.get("if_residule", False)),
        )
        if self.variant in ATTN_CORR_VARIANTS and build_attn is not None:
            model = build_attn(coord_reduction=int(self.train.get("attn_corr", {}).get("coord_reduction", 8)), **common)
        elif self.variant == "attention_phys_e1":
            e1 = self.train.get("echo1_real_only", {})
            model = build_img(
                anchor_echo_index=int(e1.get("echo_index", 0)),
                echo1_real_head=bool(e1.get("use_decoder_head", True)),
                echo1_head_filters=int(e1.get("head_filters", p.get("filters_root", 32))),
                echo1_head_depth=int(e1.get("head_depth", 2)),
                echo1_head_residual=bool(e1.get("head_residual", True)),
                **common,
            )
        else:
            model = build_img(**common)
        model.load_weights(self.weights_path)
        return model

    def predict(self, vol_hwse: np.ndarray) -> np.ndarray:
        """Correct the FULL volume; return complex [H,W,S,E] in native scale."""
        nxye = P.swap_to_nxye(vol_hwse)                       # [S,H,W,E]
        off, scale = P.norm_params(nxye, self.norm)
        norm_nxye = (nxye - off) / (scale + 1e-12)
        if self.pad192:
            norm_nxye = P.pad_width_to(norm_nxye, 192)
        x = P.to_complex_lastdim(norm_nxye)                  # [S,H,W,E,2] -- ALL slices
        if self._model is None:
            self._model = self._build(tuple(int(v) for v in x.shape[1:]))
        pred = self._model.predict(x, batch_size=self.batch_size, verbose=0).astype(np.float32)
        pred_complex = P.from_complex_lastdim(pred)          # [S,H,W,E]
        # restore native width if we padded
        if self.pad192 and pred_complex.shape[2] != nxye.shape[2]:
            pred_complex = pred_complex[:, :, : nxye.shape[2], :]
        pred_complex = P.denormalize(pred_complex, off, scale).astype(np.complex64)
        return P.swap_to_hwse(pred_complex)                  # [H,W,S,E]
