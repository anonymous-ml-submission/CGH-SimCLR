#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/plot_robustness_across_datasets.py

Plot robustness curves from one or many `robustness_summary.csv` files.

- Supports globs + $HOME + ~ + directories.
- Optional inference of missing `dataset` / `method` from file paths.
- Aggregates mean±std across seeds.
- Outputs:
  * plots/robustness_merged.csv
  * plots/robustness_knn_accuracy.png
  * plots/robustness_linear_probe_accuracy.png

Example:
  python tools/plot_robustness_across_datasets.py \
    --inputs "./runs/**/robustness_summary.csv" \
    --outdir plots --out_csv plots/robustness_merged.csv \
    --seeds 0 1 2 3 4 \
    --infer_dataset_method

By default, this script does NOT draw shaded bands (to avoid a 'colored strip' look).
Use --shade if you want mean±std bands.
"""

import argparse
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def expand_inputs(patterns):
    files = []
    for pat in patterns:
        pat = os.path.expandvars(os.path.expanduser(pat))
        if os.path.isdir(pat):
            pat = os.path.join(pat, "**", "robustness_summary.csv")
        files.extend(glob.glob(pat, recursive=True))
    files = [f for f in sorted(set(files)) if os.path.isfile(f)]
    return files


def infer_dataset(path_str: str):
    s = path_str.lower()
    for ds in ("cifar10", "cifar100", "stl10"):
        if ds in s:
            return ds
    return None


def infer_method(path_str: str):
    s = path_str.lower()
    if "baseline" in s:
        return "baseline"
    if "cgh" in s:
        return "cgh"
    return None


def std_ddof1(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Paths or globs, e.g. "./runs/**/robustness_summary.csv"")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--out_csv", default="plots/robustness_merged.csv")

    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--ps", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)

    ap.add_argument("--infer_dataset_method", action="store_true",
                    help="Infer missing dataset/method columns from file path.")
    ap.add_argument("--shade", action="store_true",
                    help="Draw mean±std bands (off by default).")
    args = ap.parse_args()

    files = expand_inputs(args.inputs)
    if not files:
        raise SystemExit("No robustness_summary.csv files matched --inputs. Check quoting/pattern.")

    dfs = []
    for fp in files:
        d = pd.read_csv(fp)
        d.columns = [c.strip() for c in d.columns]
        d["source"] = fp

        # normalize
        if "dataset" in d.columns:
            d["dataset"] = d["dataset"].astype(str).str.lower().str.strip()
        if "method" in d.columns:
            d["method"] = d["method"].astype(str).str.lower().str.strip()
        if "pos_corrupt_p" not in d.columns and "p" in d.columns:
            d = d.rename(columns={"p": "pos_corrupt_p"})

        for c in ["seed","epochs","gamma","k_gate","pos_corrupt_p","knn_acc","linear_acc"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")

        if args.infer_dataset_method:
            ds = infer_dataset(fp)
            md = infer_method(fp)
            if ("dataset" not in d.columns) or d["dataset"].isna().all():
                if ds is not None:
                    d["dataset"] = ds
            if ("method" not in d.columns) or d["method"].isna().all():
                if md is not None:
                    d["method"] = md

        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)

    # filters
    if args.datasets is not None and "dataset" in df.columns:
        keep = {x.lower() for x in args.datasets}
        df = df[df["dataset"].isin(keep)].copy()
    if args.methods is not None and "method" in df.columns:
        keep = {x.lower() for x in args.methods}
        df = df[df["method"].isin(keep)].copy()
    if args.ps is not None and "pos_corrupt_p" in df.columns:
        keep = {float(x) for x in args.ps}
        df = df[df["pos_corrupt_p"].isin(keep)].copy()
    if args.seeds is not None and "seed" in df.columns:
        df = df[df["seed"].isin(set(args.seeds))].copy()

    # de-dup (keep last per config+seed)
    dedup_cols = [c for c in ["dataset","method","pos_corrupt_p","gamma","k_gate","arch","epochs","seed"] if c in df.columns]
    if "time" in df.columns:
        df = df.sort_values("time")
    df = df.drop_duplicates(subset=dedup_cols, keep="last")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("[WROTE]", out_csv)

    need = ["dataset","method","pos_corrupt_p","seed","knn_acc","linear_acc"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns for plotting: {missing}")

    g = (
        df.groupby(["dataset","method","pos_corrupt_p"], as_index=False, dropna=False)
          .agg(
              knn_mean=("knn_acc","mean"),
              knn_std=("knn_acc", std_ddof1),
              lin_mean=("linear_acc","mean"),
              lin_std=("linear_acc", std_ddof1),
              n_seeds=("seed","nunique"),
          )
          .sort_values(["dataset","method","pos_corrupt_p"])
    )

    datasets_order = ["cifar10","cifar100","stl10"]
    methods_order = ["cgh","baseline"]
    label_map = {"cgh":"CGH", "baseline":"SimCLR"}

    def plot_metric(mean_col, std_col, ylabel, fname):
        plt.figure()
        for ds in datasets_order:
            for m in methods_order:
                gd = g[(g["dataset"] == ds) & (g["method"] == m)].copy()
                if gd.empty:
                    continue
                x = gd["pos_corrupt_p"].to_numpy(dtype=float)
                y = (gd[mean_col].to_numpy(dtype=float) * 100.0)
                s = (gd[std_col].fillna(0.0).to_numpy(dtype=float) * 100.0)

                lbl = f"{ds.upper()}-{label_map.get(m, m)}"
                plt.plot(x, y, marker="o", label=lbl)
                if args.shade:
                    plt.fill_between(x, y - s, y + s, alpha=0.15, linewidth=0)

        plt.xlabel("Positive-pair corruption probability p")
        plt.ylabel(ylabel)
        if len(g["pos_corrupt_p"].dropna().unique()) > 0:
            plt.xticks(sorted(g["pos_corrupt_p"].dropna().unique()))
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        out = outdir / fname
        plt.savefig(out, dpi=200)
        plt.close()
        print("[WROTE]", out)

    plot_metric("knn_mean", "knn_std", "kNN accuracy (%)", "robustness_knn_accuracy.png")
    plot_metric("lin_mean", "lin_std", "Linear probe accuracy (%)", "robustness_linear_probe_accuracy.png")


if __name__ == "__main__":
    main()
