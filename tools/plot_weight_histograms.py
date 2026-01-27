#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/plot_weight_histograms.py

Plot weight histogram(s) from one or many w_hist.csv files (produced by scripts/train.py).

Modes
-----
1) line   : your original plot (hist mass vs weight bin center), averaged across seeds,
           one curve per dataset for a fixed (method, p, epoch).
2) heatmap: epoch x bin heatmap per (dataset, method, p, gamma, k_gate, seed).

Examples
--------
# (A) From repo root, use local runs/
python tools/plot_weight_histograms.py --inputs runs/w_hist.csv --mode line --method cgh --p 0.0 --epoch 1 --outdir plots

# (B) From anywhere, point to another tree
python /path/to/icml_cgh_simclr_artifact/tools/plot_weight_histograms.py \
  --inputs "./runs/**/w_hist.csv" \
  --mode line --method cgh --p 0.0 --epoch 1 --outdir /tmp/plots
"""

import argparse, glob, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def expand_inputs(patterns):
    files = []
    for pat in patterns:
        pat = os.path.expandvars(os.path.expanduser(pat))
        files.extend(glob.glob(pat, recursive=True))
    return sorted(set(files))


def sample_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def infer_bin_cols(df):
    bin_cols = [c for c in df.columns if str(c).startswith("bin")]
    if not bin_cols:
        raise ValueError("No binXX columns found in w_hist.csv")
    # stable ordering: bin00, bin01, ...
    try:
        bin_cols = sorted(bin_cols, key=lambda c: int("".join(ch for ch in c if ch.isdigit()) or -1))
    except Exception:
        bin_cols = sorted(bin_cols)
    return bin_cols


def infer_centers(df, n_bins):
    # best: use wmin/wmax if present
    if "wmin" in df.columns and "wmax" in df.columns and df["wmin"].notna().any() and df["wmax"].notna().any():
        wmin = float(df["wmin"].dropna().iloc[0])
        wmax = float(df["wmax"].dropna().iloc[0])
        edges = np.linspace(wmin, wmax, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers
    # fallback: just 0..n_bins-1 (still works)
    return np.arange(n_bins, dtype=float)


def plot_line(df, outdir: Path, method: str, p: float, epoch: int, seeds: str, shade: bool, title_prefix: str):
    bin_cols = infer_bin_cols(df)
    centers = infer_centers(df, len(bin_cols))

    d = df.copy()
    d["dataset"] = d["dataset"].astype(str).str.lower().str.strip() if "dataset" in d.columns else "na"
    d["method"] = d["method"].astype(str).str.lower().str.strip() if "method" in d.columns else "na"

    if "pos_corrupt_p" not in d.columns and "p" in d.columns:
        d = d.rename(columns={"p": "pos_corrupt_p"})

    for c in ["pos_corrupt_p", "epoch", "seed"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    method = method.lower()
    seed_set = set(int(s) for s in seeds.split(",") if s.strip() != "")

    # filter
    if "method" in d.columns:
        d = d[d["method"] == method]
    if "pos_corrupt_p" in d.columns:
        d = d[d["pos_corrupt_p"] == float(p)]
    if "epoch" in d.columns:
        d = d[d["epoch"] == int(epoch)]
    if "seed" in d.columns and seed_set:
        d = d[d["seed"].isin(seed_set)]

    if d.empty:
        raise ValueError(f"No rows after filtering: method={method}, p={p}, epoch={epoch}. Check inputs/columns.")

    plt.figure()
    for ds, g in d.groupby("dataset", dropna=False):
        M = g[bin_cols].to_numpy(dtype=float)   # (seeds, bins) possibly >1 rows per seed if multiple sources
        mu = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(mu)

        plt.plot(centers, mu, marker="o", label=str(ds).upper())
        if shade and M.shape[0] > 1:
            plt.fill_between(centers, mu - sd, mu + sd, alpha=0.15)

    plt.xlabel("Weight w")
    plt.ylabel("Probability mass (hist)")
    title = f"{title_prefix} {method.upper()} weight histogram   p={p:g}   epoch={epoch}"
    plt.title(title.strip())
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"whist_ALL_p{p:g}_ep{epoch}_{method}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("[WROTE]", out)


def plot_heatmaps(df, outdir: Path, query: str):
    bin_cols = infer_bin_cols(df)
    d = df.copy()
    if query.strip():
        d = d.query(query)

    outdir.mkdir(parents=True, exist_ok=True)

    keys = [c for c in ["dataset", "method", "pos_corrupt_p", "gamma", "k_gate", "seed"] if c in d.columns]
    if not keys:
        keys = ["method"] if "method" in d.columns else []

    for g, gg in d.groupby(keys, dropna=False):
        g = g if isinstance(g, tuple) else (g,)
        tag = "_".join([f"{k}={v}" for k, v in zip(keys, g)]).replace("/", "-").replace(" ", "")
        gg = gg.sort_values("epoch" if "epoch" in gg.columns else gg.index)

        mat = gg[bin_cols].to_numpy(dtype=float)  # (epochs, bins)
        plt.figure()
        plt.imshow(mat, aspect="auto", origin="lower")
        plt.xlabel("bin")
        plt.ylabel("epoch_index")
        plt.title(tag)
        plt.colorbar()
        plt.tight_layout()

        out = outdir / f"w_hist_{tag}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print("[WROTE]", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=["runs/**/w_hist.csv"], help="CSV paths or globs (supports **).")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--mode", choices=["line", "heatmap"], default="line")

    # line-mode args (your original)
    ap.add_argument("--method", default="cgh", help="cgh or baseline/simclr etc.")
    ap.add_argument("--p", type=float, default=0.0, help="pos_corrupt_p value")
    ap.add_argument("--epoch", type=int, default=1, help="epoch number")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--shade", action="store_true", help="Enable ±std shading (default: off).")
    ap.add_argument("--title_prefix", default="")

    # heatmap-mode args
    ap.add_argument("--filter", default="", help="Pandas query for heatmap mode, e.g. dataset=='cifar10' and method=='cgh'")
    args = ap.parse_args()

    files = expand_inputs(args.inputs)
    if not files:
        raise SystemExit("No w_hist.csv files matched --inputs.")

    dfs = []
    for fp in files:
        d = pd.read_csv(fp)
        d.columns = [c.strip() for c in d.columns]
        d["__source__"] = fp
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)

    outdir = Path(os.path.expandvars(os.path.expanduser(args.outdir)))

    if args.mode == "line":
        plot_line(
            df=df,
            outdir=outdir,
            method=args.method,
            p=args.p,
            epoch=args.epoch,
            seeds=args.seeds,
            shade=args.shade,
            title_prefix=args.title_prefix,
        )
    else:
        plot_heatmaps(df=df, outdir=outdir, query=args.filter)


if __name__ == "__main__":
    main()
