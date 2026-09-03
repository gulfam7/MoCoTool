"""MAT I/O for MoCoTool. Handles v7 (scipy) and v7.3 (HDF5), auto-detects the
image variable name, and preserves it on save."""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple

import numpy as np

PREFERRED_KEYS = ("ima_comb", "ima_comb1", "pre")


def load_volume(path: str, preferred_keys: Iterable[str] = PREFERRED_KEYS) -> Tuple[np.ndarray, str]:
    """Return (complex4d_volume [H,W,S,E], variable_name)."""
    try:
        from scipy.io import loadmat

        mat = loadmat(path, verify_compressed_data_integrity=False)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        name = _pick_key(keys, preferred_keys, path)
        return np.asarray(mat[name]), name
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as f:
            keys = [k for k in f.keys() if not k.startswith("#")]
            name = _pick_key(keys, preferred_keys, path)
            raw = f[name][()]
            # h5py returns compound dtype for complex MAT v7.3
            if raw.dtype.names and "real" in raw.dtype.names:
                arr = raw["real"] + 1j * raw["imag"]
            else:
                arr = np.asarray(raw)
            # MATLAB v7.3 stores transposed; restore [H,W,S,E]
            arr = np.transpose(arr, tuple(range(arr.ndim))[::-1])
            return np.asarray(arr), name


def _pick_key(keys, preferred, path):
    for k in preferred:
        if k in keys:
            return k
    four_d = [k for k in keys if k != "fix_info"]
    if len(four_d) == 1:
        return four_d[0]
    raise KeyError(f"Could not identify the image variable in {path}. Keys: {sorted(keys)}")


def save_volume(path: str, volume: np.ndarray, varname: str, extra: Dict[str, Any] | None = None) -> None:
    """Save a complex [H,W,S,E] volume under `varname` (native scale)."""
    from scipy.io import savemat

    payload: Dict[str, Any] = {varname: np.asarray(volume, dtype=np.complex64)}
    if extra:
        payload.update(extra)
    savemat(path, payload, do_compression=True)


def subject_id_from_path(path: str, prefix: str = "ima_comb_") -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[len(prefix):] if stem.startswith(prefix) else stem
