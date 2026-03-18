"""Evaluate SPADE on a benchmark dataset and write metrics to CSV.

Wraps vendor/SPADE/evaluate.py's eval_model() so that results are saved
as a structured CSV instead of just printed to the terminal.

Usage:
    python -m src.spade.run_eval \\
        --dataset flsea_demo \\
        --weights /path/to/underwater_depth_pipeline.pt

    python -m src.spade.run_eval \\
        --dataset seathru \\
        --weights /path/to/weights.pt \\
        --filenames_file /abs/path/to/data/spade_lists/seathru_test.txt \\
        --save_image

Per-dataset defaults are read from configs/spade_datasets.yaml.
CLI flags override the YAML values.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT    = Path(__file__).resolve().parents[2]
VENDOR_SPADE = REPO_ROOT / "vendor" / "SPADE"
SPADE_CFG    = REPO_ROOT / "configs" / "spade_datasets.yaml"


def _load_dataset_config(tag: str) -> dict:
    if SPADE_CFG.exists():
        with open(SPADE_CFG) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("datasets", {}).get(tag, {})
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run SPADE evaluation and save metrics to CSV."
    )
    ap.add_argument(
        "--dataset", required=True,
        help="Dataset tag, e.g. flsea_demo, seathru, kskin",
    )
    ap.add_argument(
        "--weights", required=True,
        help="Absolute path to .pt pretrained weights file",
    )
    ap.add_argument(
        "--filenames_file", default=None,
        help="Override: absolute path to filenames list. "
             "Default comes from configs/spade_datasets.yaml.",
    )
    ap.add_argument(
        "--ranges", nargs="+", type=float, default=None,
        help="Evaluation depth thresholds in metres (default: from dataset config)",
    )
    ap.add_argument(
        "--out_csv", default=None,
        help="Output CSV path (default: reports/spade/<dataset>_metrics.csv)",
    )
    ap.add_argument(
        "--save_image", action="store_true",
        help="Save depth visualisations alongside the CSV",
    )
    ap.add_argument(
        "--output_image_path", default=None,
        help="Directory for depth images (default: reports/spade/<dataset>/)",
    )
    args = ap.parse_args()

    ds_cfg = _load_dataset_config(args.dataset)

    # ── Resolve all paths before os.chdir ────────────────────────────────────
    weights = str(Path(args.weights).resolve())

    out_csv = (
        Path(args.out_csv).resolve() if args.out_csv
        else REPO_ROOT / "reports" / "spade" / f"{args.dataset}_metrics.csv"
    )
    img_out = (
        str(Path(args.output_image_path).resolve()) if args.output_image_path
        else str(REPO_ROOT / "reports" / "spade" / args.dataset)
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.save_image:
        Path(img_out).mkdir(parents=True, exist_ok=True)

    # ── Build eval_model kwargs from YAML + CLI overrides ────────────────────
    kwargs: dict = {}

    # filenames list: CLI arg > YAML > (vendor default for flsea_demo)
    raw_ff = args.filenames_file or ds_cfg.get("filenames_file_eval")
    if raw_ff:
        kwargs["filenames_file_eval"] = str(Path(raw_ff).resolve())

    # data path prefixes
    data_path = ds_cfg.get("data_path_eval", "")
    gt_path   = ds_cfg.get("gt_path_eval", "")
    if data_path:
        kwargs["data_path_eval"] = data_path
    if gt_path:
        kwargs["gt_path_eval"] = gt_path

    kwargs["min_depth"] = ds_cfg.get("min_depth", 0.1)
    kwargs["max_depth"] = ds_cfg.get("max_depth", 18.0)

    ranges = args.ranges or ds_cfg.get("eval_ranges", [10, 5, 2])

    # ── Import SPADE (must run from vendor/SPADE/ for UnderwaterDepth.* imports)
    sys.path.insert(0, str(VENDOR_SPADE))
    os.chdir(VENDOR_SPADE)

    from evaluate import eval_model  # noqa: E402

    print(f"=== SPADE evaluation: {args.dataset} | ranges {ranges} m ===")
    metrics_list = eval_model(
        "SPADE",
        pretrained_resource=f"local::{weights}",
        dataset="flsea_sparse_feature",
        ranges=ranges,
        save_image=args.save_image,
        output_image_path=img_out,
        **kwargs,
    )

    # ── Write CSV ─────────────────────────────────────────────────────────────
    rows = []
    for rng, metrics in zip(ranges, metrics_list):
        row = {"dataset": args.dataset, "range_m": rng}
        row.update(metrics)
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Metrics saved → {out_csv}")
    else:
        print("WARNING: no metrics computed — check that depth GT is available.")


if __name__ == "__main__":
    main()
