#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/merge_robustness_results.py

Merge robustness_summary.csv files and print seed-aggregated summaries.
Reviewer-friendly: no hardcoded paths, supports globs + $HOME + ~.
"""

import argparse
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _expand_patterns(patterns):
    files = []
    for pat in patterns:
        pat = os.path.expandvars(os.path.expandvars(os.path.expanduser(pat)))
        # If pattern contains glob chars -> expand, else treat as literal file
        if any(ch in pat for ch in ["*", "?", "[", "]"]):
            files.extend(glob.glob(pat, recursive=True))
        else:
            files.append(pat)
    # keep only existing files
    files = [Path(f) for f in files if Path(f).exists()]
    # unique + sorted
    return sorted({f.resolve() for f in files})


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # standardize common fields if present
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str).str.lower().str.strip()
    if "method" in df.columns:
        df["method"] = df["method"].astype(str).str.lower().str.strip()

    # allow either pos_corrupt_p or p
    if "pos_corrupt_p" not in df.columns and "p" in df.columns:
        df = df.rename(columns={"p": "pos_corrupt_p"})

    # numeric coercions
    for c in ["seed", "epochs", "gamma", "k_gate", "pos_corrupt_p", "knn_acc", "linear_acc"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _fmt_pct(mean, std):
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        std = 0.0
    return f"{mean*100:.2f} ± {std*100:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=["runs/**/robustness_summary.csv"],
        help="One or more CSV paths or globs. Example: runs/**/robustness_summary.csv"
    )
    ap.add_argument("--out_csv", default="plots/robustness_merged.csv")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Optional list like: cgh baseline (default: keep all).")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Optional list like: cifar10 cifar100 stl10 (default: keep all).")
    ap.add_argument("--ps", nargs="*", default=None,
                    help="Optional p values like: 0 0.2 0.3 0.4 (default: keep all).")
    args = ap.parse_args()

    files = _expand_patterns(args.inputs)
    if not files:
        raise SystemExit("No robustness_summary.csv files matched --inputs.")

    dfs = []
    for fp in files:
        d = pd.read_csv(fp)
        d = _normalize_cols(d)
        d["source"] = str(fp)
        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)

    # optional filters
    if args.methods is not None and "method" in df.columns:
        keep = {m.lower() for m in args.methods}
        df = df[df["method"].isin(keep)].copy()

    if args.datasets is not None and "dataset" in df.columns:
        keep = {d.lower() for d in args.datasets}
        df = df[df["dataset"].isin(keep)].copy()

    if args.ps is not None and "pos_corrupt_p" in df.columns:
        keep = set(float(x) for x in args.ps)
        df = df[df["pos_corrupt_p"].isin(keep)].copy()

    # de-duplicate repeated exports (keep last per config+seed)
    dedup_cols = [c for c in ['dataset','method','pos_corrupt_p','gamma','k_gate','arch','epochs','seed'] if c in df.columns]
    if 'time' in df.columns:
        df = df.sort_values('time')
    df = df.drop_duplicates(subset=dedup_cols, keep='last')

    # write merged CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # -------- summary (mean±std over seeds) --------
    # group columns that define an experimental config
    cfg_cols = [c for c in ["dataset", "method", "pos_corrupt_p", "gamma", "k_gate", "arch", "epochs"] if c in df.columns]

    if not cfg_cols:
        print("\n[WARN] No standard config columns found; wrote merged CSV only.")
        print("[WROTE]", out_csv)
        return

    # robust seed counting
    have_seed = "seed" in df.columns

    agg = (
        df.groupby(cfg_cols, dropna=False)
          .agg(
              knn_mean=("knn_acc", "mean"),
              knn_std=("knn_acc", "std"),
              lin_mean=("linear_acc", "mean"),
              lin_std=("linear_acc", "std"),
              n_rows=("knn_acc", "count"),
              n_seeds=("seed", pd.Series.nunique) if have_seed else ("knn_acc", "count"),
          )
          .reset_index()
    )

    sort_cols = [c for c in ["dataset", "method", "pos_corrupt_p", "gamma", "k_gate"] if c in agg.columns]
    if sort_cols:
        agg = agg.sort_values(sort_cols)

    # pretty print
    show = agg.copy()
    show["knn(mean±std)%"] = [
        _fmt_pct(m, s) for m, s in zip(show["knn_mean"].to_numpy(), show["knn_std"].to_numpy())
    ]
    show["lin(mean±std)%"] = [
        _fmt_pct(m, s) for m, s in zip(show["lin_mean"].to_numpy(), show["lin_std"].to_numpy())
    ]

    keep_cols = [c for c in cfg_cols if c in show.columns] + ["n_seeds", "knn(mean±std)%", "lin(mean±std)%"]
    show = show[keep_cols]

    print(f"\n[INFO] loaded {len(files)} files | merged rows={len(df)}")
    print("[SUMMARY] mean±std over seeds")
    with pd.option_context("display.max_rows", 300, "display.max_columns", 80):
        print(show)

    print("\n[WROTE]", out_csv)


if __name__ == "__main__":
    main()
