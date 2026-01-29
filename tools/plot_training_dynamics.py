#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/plot_dynamics_across_datasets.py

Training dynamics plots from one or many dynamics.csv files.

Generates TWO plot families (CGH-only by default):
A) Across datasets at fixed p:   dataset lines (CIFAR10/CIFAR100/STL10), mean±std over seeds
B) Per dataset at fixed metric:  p lines (p=0.0,0.2,0.3,0.4...), mean±std over seeds

Typical usage:
  python tools/plot_dynamics_across_datasets.py \
    --inputs "./runs/**/dynamics.csv" \
    --outdir plots \
    --metrics corr_w_h_mean ESS_mean gate_on_frac_mean \
    --methods cgh \
    --seeds 0,1,2,3,4

Notes:
- Uses python glob (recursive) and expands $HOME and ~.
- Robust to missing 'dataset' column if you provide --forced_dataset for each file pattern (optional).
"""

import argparse
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATASET_ORDER_DEFAULT = ["cifar10", "cifar100", "stl10"]


def ddof1_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def expand_patterns(patterns):
    out = []
    for pat in patterns:
        pat = os.path.expanduser(os.path.expandvars(pat))
        out.extend(glob.glob(pat, recursive=True))
    return sorted(set(out))


def normalize_df(d: pd.DataFrame) -> pd.DataFrame:
    d.columns = [c.strip() for c in d.columns]

    # allow either pos_corrupt_p or p
    if "pos_corrupt_p" not in d.columns and "p" in d.columns:
        d = d.rename(columns={"p": "pos_corrupt_p"})

    # normalize string cols
    if "dataset" in d.columns:
        d["dataset"] = d["dataset"].astype(str).str.lower().str.strip()
    if "method" in d.columns:
        d["method"] = d["method"].astype(str).str.lower().str.strip()

    # numeric cols
    for c in ["epoch", "seed", "pos_corrupt_p", "gamma", "k_gate"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    return d


def ensure_required_cols(df: pd.DataFrame):
    need = ["epoch", "pos_corrupt_p", "method"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"dynamics.csv missing required columns: {miss}. Found columns: {df.columns.tolist()}")


def agg_mean_std(df: pd.DataFrame, group_cols, metric: str) -> pd.DataFrame:
    # mean±std over seeds at each epoch (std ddof=1)
    return (
        df.groupby(group_cols, as_index=False)[metric]
          .agg(mean="mean", std=ddof1_std)
    )


def plot_across_datasets(df, outdir: Path, metrics, dataset_order, seeds, methods, title_prefix="CGH dynamics across datasets"):
    """
    For each p and metric:
      one plot with dataset lines (CIFAR10/CIFAR100/STL10), mean±std over seeds
    """
    # filter seeds
    if "seed" in df.columns and seeds is not None:
        df = df[df["seed"].isin(seeds)].copy()

    # filter methods
    if methods is not None:
        df = df[df["method"].isin(methods)].copy()

    if "dataset" not in df.columns:
        raise ValueError("No 'dataset' column found. Across-dataset plots need dataset labels.")

    ps = sorted([p for p in df["pos_corrupt_p"].dropna().unique()])

    for p in ps:
        d0 = df[df["pos_corrupt_p"] == p].copy()
        if d0.empty:
            continue

        for metric in metrics:
            if metric not in d0.columns:
                continue

            d = d0[["epoch", "dataset", metric] + (["seed"] if "seed" in d0.columns else [])].copy()
            d[metric] = pd.to_numeric(d[metric], errors="coerce")
            d = d[d[metric].notna()].copy()
            if d.empty:
                continue

            # group_cols: epoch + dataset (+ seed implicit inside mean)
            if "seed" in d.columns:
                g = agg_mean_std(d, ["epoch", "dataset"], metric).sort_values(["dataset", "epoch"])
            else:
                # if no seed column, just treat values as already-aggregated
                g = d.groupby(["epoch", "dataset"], as_index=False)[metric].mean()
                g = g.rename(columns={metric: "mean"})
                g["std"] = 0.0
                g = g.sort_values(["dataset", "epoch"])

            plt.figure()
            for ds in dataset_order:
                gm = g[g["dataset"] == ds]
                if gm.empty:
                    continue
                x = gm["epoch"].to_numpy()
                y = gm["mean"].to_numpy()
                s = gm["std"].fillna(0).to_numpy()
                plt.plot(x, y, label=ds.upper())
# =============================================================================
#                 plt.fill_between(x, y - s, y + s, alpha=0.15)
# =============================================================================

            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title(f"{title_prefix}   p={p:g}   {metric}")
            plt.grid(True, alpha=0.3)
            plt.legend(ncol=2)

            out = outdir / f"dynamics_ALL_p{p:g}_{metric}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()
            print("[WROTE]", out)


def plot_per_dataset(df, outdir: Path, metrics, seeds, methods, dataset_order, title_prefix="Dynamics"):
    """
    For each dataset and metric:
      one plot with p-lines (p=0.0,0.2,0.3...), mean±std over seeds
    Matches your "Dynamics CIFAR10 corr_w_cr_mean" style.
    """
    # filter seeds
    if "seed" in df.columns and seeds is not None:
        df = df[df["seed"].isin(seeds)].copy()

    # filter methods (keep only CGH unless user asks otherwise)
    if methods is not None:
        df = df[df["method"].isin(methods)].copy()

    if "dataset" not in df.columns:
        raise ValueError("No 'dataset' column found. Per-dataset plots need dataset labels.")

    for ds in dataset_order:
        d0 = df[df["dataset"] == ds].copy()
        if d0.empty:
            continue

        ps = sorted([p for p in d0["pos_corrupt_p"].dropna().unique()])
        for metric in metrics:
            if metric not in d0.columns:
                continue
            d = d0[["epoch", "pos_corrupt_p", metric] + (["seed"] if "seed" in d0.columns else [])].copy()
            d[metric] = pd.to_numeric(d[metric], errors="coerce")
            d = d[d[metric].notna()].copy()
            if d.empty:
                continue

            if "seed" in d.columns:
                g = agg_mean_std(d, ["epoch", "pos_corrupt_p"], metric).sort_values(["pos_corrupt_p", "epoch"])
            else:
                g = d.groupby(["epoch", "pos_corrupt_p"], as_index=False)[metric].mean()
                g = g.rename(columns={metric: "mean"})
                g["std"] = 0.0
                g = g.sort_values(["pos_corrupt_p", "epoch"])

            plt.figure()
            for p in ps:
                gp = g[g["pos_corrupt_p"] == p]
                if gp.empty:
                    continue
                x = gp["epoch"].to_numpy()
                y = gp["mean"].to_numpy()
                s = gp["std"].fillna(0).to_numpy()
                plt.plot(x, y, label=f"cgh p={p:g}")
                plt.fill_between(x, y - s, y + s, alpha=0.15)

            plt.xlabel("epoch")
            plt.ylabel(metric)
            plt.title(f"{title_prefix}  {ds.upper()}  {metric}")
            plt.legend(loc="lower left")
            plt.tight_layout()

            out = outdir / f"dynamics_{ds}_{metric}.png"
            plt.savefig(out, dpi=200)
            plt.close()
            print("[WROTE]", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more paths/globs to dynamics.csv. Example: ./runs/**/dynamics.csv")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--metrics", nargs="+", default=["ESS_mean", "corr_w_h_mean", "gate_on_frac_mean"],
                    help="Metric columns to plot.")
    ap.add_argument("--methods", nargs="+", default=["cgh"],
                    help="Which methods to include (default: cgh only). Use: --methods cgh baseline")
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="Comma-separated seeds to average (default: 0,1,2,3,4). Use empty string to disable filtering.")
    ap.add_argument("--dataset_order", default="cifar10,cifar100,stl10",
                    help="Comma-separated dataset order for plotting.")
    ap.add_argument("--no_across", action="store_true", help="Disable across-datasets plots.")
    ap.add_argument("--no_per_dataset", action="store_true", help="Disable per-dataset plots.")
    args = ap.parse_args()

    files = expand_patterns(args.inputs)
    if not files:
        raise SystemExit("No dynamics.csv files matched --inputs. Check quoting and $HOME expansion.")

    seeds = None
    if args.seeds.strip() != "":
        seeds = set(int(s) for s in args.seeds.split(",") if s.strip() != "")

    dataset_order = [s.strip().lower() for s in args.dataset_order.split(",") if s.strip() != ""]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for fp in files:
        d = pd.read_csv(fp)
        d = normalize_df(d)
        d["__source__"] = fp
        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    ensure_required_cols(df)

    # If method filter provided, apply early (reduces baseline noise)
    if args.methods:
        df = df[df["method"].isin([m.lower() for m in args.methods])].copy()

    # Make sure epoch is numeric
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df[df["epoch"].notna()].copy()

    # INFO
    print(f"[INFO] loaded {len(files)} files, total rows={len(df)}")
    print("[INFO] methods:", sorted(df["method"].dropna().unique().tolist()))
    if "dataset" in df.columns:
        print("[INFO] datasets:", sorted(df["dataset"].dropna().unique().tolist()))
    print("[INFO] p values:", sorted(df["pos_corrupt_p"].dropna().unique().tolist()))

    if not args.no_across:
        plot_across_datasets(
            df=df, outdir=outdir, metrics=args.metrics,
            dataset_order=dataset_order, seeds=seeds,
            methods=[m.lower() for m in args.methods] if args.methods else None
        )

    if not args.no_per_dataset:
        plot_per_dataset(
            df=df, outdir=outdir, metrics=args.metrics,
            seeds=seeds, methods=[m.lower() for m in args.methods] if args.methods else None,
            dataset_order=dataset_order
        )


if __name__ == "__main__":
    main()
