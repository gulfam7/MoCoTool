#!/usr/bin/env python3
"""MoCoTool command-line interface (headless).

Verifies the engine end-to-end before any UI exists. Single file or batch folder,
three modes, optional gate override and threshold.

Examples
--------
  # single, gated detect+correct loop (max 3 iters)
  python cli.py single --input "C:/data/ima_comb_110.mat" --mode detect_correct --output_dir out

  # batch detection only over a folder
  python cli.py batch --input_dir "C:/data/motion" --mode detect --output_dir out

  # batch correction, forced (ignore the gate)
  python cli.py batch --input_dir "C:/data/motion" --mode correct --output_dir out --force_correct
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_TOOL_ROOT = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_TOOL_ROOT, ".."))
if _TOOL_ROOT not in sys.path:
    sys.path.insert(0, _TOOL_ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from engine import load_engine                    # noqa: E402
from engine.reporting import write_summary, clean_for_json  # noqa: E402

INPUT_GLOB = "ima_comb_*.mat"


def _gate_override(args):
    if getattr(args, "force_correct", False):
        return True
    if getattr(args, "skip_correct", False):
        return False
    return None


def run_single(args):
    engine = load_engine(_TOOL_ROOT, _REPO_ROOT, max_iters=args.max_iters)
    res = engine.process_volume(
        args.input, args.mode, output_dir=args.output_dir,
        force_correct=_gate_override(args), threshold=args.threshold,
        save_output=(args.mode != "detect"),
        progress=lambda m: print(f"  [{res_id(args.input)}] {m}"),
    )
    print(json.dumps(clean_for_json(res), indent=2, default=str))
    return 0 if res["status"] == "ok" else 1


def run_batch(args):
    files = sorted(glob.glob(os.path.join(args.input_dir, INPUT_GLOB)))
    files = [f for f in files if "_corrected" not in f and "_fix192" not in f]
    if not files:
        print(f"No {INPUT_GLOB} files in {args.input_dir}")
        return 1
    print(f"Found {len(files)} subject(s).")
    engine = load_engine(_TOOL_ROOT, _REPO_ROOT, max_iters=args.max_iters)

    results = []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.basename(f)}")
        res = engine.process_volume(
            f, args.mode, output_dir=args.output_dir,
            force_correct=_gate_override(args), threshold=args.threshold,
            save_output=(args.mode != "detect"),
            progress=lambda m: print(f"    {m}"),
        )
        dec = res.get("decision", "-")
        corr = "corrected" if res.get("corrected") else "not corrected"
        print(f"    -> {res['status']} | {dec} | {corr} | {res.get('seconds','?')}s")
        results.append(clean_for_json(res))

    os.makedirs(args.output_dir, exist_ok=True)
    summary = os.path.join(args.output_dir, "batch_summary.xlsx")
    write_summary(summary, results)
    n_corr = sum(1 for r in results if r.get("corrected"))
    n_err = sum(1 for r in results if r.get("status") == "error")
    print(f"\nSummary -> {summary}")
    print(f"Totals: {len(results)} subjects, corrected={n_corr}, errors={n_err}")
    return 0


def res_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def build_parser():
    ap = argparse.ArgumentParser(description="MoCoTool CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mode", choices=["detect", "correct", "detect_correct"], default="detect_correct")
    common.add_argument("--output_dir", default="mocotool_output")
    common.add_argument("--threshold", type=float, default=None, help="override detection threshold")
    common.add_argument("--max_iters", type=int, default=3)
    common.add_argument("--force_correct", action="store_true", help="always correct (ignore gate)")
    common.add_argument("--skip_correct", action="store_true", help="never correct (detection only path)")

    ps = sub.add_parser("single", parents=[common]); ps.add_argument("--input", required=True); ps.set_defaults(func=run_single)
    pb = sub.add_parser("batch", parents=[common]); pb.add_argument("--input_dir", required=True); pb.set_defaults(func=run_batch)
    return ap


def main():
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
