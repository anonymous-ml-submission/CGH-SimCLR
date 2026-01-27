#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/format_clean_eval_table.py

Aggregate robustness_summary.csv files across seeds and print LaTeX rows
(mean ± std, in percent) for kNN and linear probe.
"""

import argparse, glob
from typing import Optional, List, Dict
import pandas as pd
import numpy as np

SEEDS_DEFAULT = "0,1,2,3,4"


def sample_std(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def format_mean_std_pct(m, s) -> str:
    if not np.isfinite(m):
        return "NA"
    if not np.isfinite(s):
        s = 0.0
    return f"{m*100:.2f} $\\pm$ {s*100:.2f}"


def normalize_columns(d: pd.DataFrame) -> pd.DataFrame:
    d.columns = [c.strip() for c in d.columns]

    if "dataset" in d.columns:
        d["dataset"] = d["dataset"].astype(str).str.lower()
    if "method" in d.columns:
        d["method"] = d["method"].astype(str).str.lower()

    for c in ["seed", "epochs", "gamma", "k_gate", "pos_corrupt_p", "knn_acc", "linear_acc"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    return d


def force_dataset_if_missing(d: pd.DataFrame, forced_ds: Optional[str]) -> pd.DataFrame:
    if forced_ds is None:
        if "dataset" not in d.columns:
            raise ValueError("Input CSV has no 'dataset' column and no forced dataset was provided.")
        return d

    if ("dataset" not in d.columns) or d["dataset"].isna().all():
        d["dataset"] = forced_ds
    return d


def ensure_gamma_kg(d: pd.DataFrame) -> pd.DataFrame:
    if "gamma" not in d.columns:
        d["gamma"] = np.nan
    if "k_gate" not in d.columns:
        d["k_gate"] = np.nan
    return d


def infer_variant(row: pd.Series) -> str:
    m = str(row.get("method", "")).lower()

    if m in ("baseline", "simclr"):
        return "SimCLR"

    g = row.get("gamma", np.nan)
    k = row.get("k_gate", np.nan)

    def close(a, b):
        return np.isfinite(a) and abs(float(a) - float(b)) < 1e-6

    if close(g, 0.0) and close(k, 0.30): return "Gate-only"
    if close(g, 0.5) and close(k, 0.30): return "Mild CGH"
    if close(g, 1.0) and close(k, 0.30): return "Full CGH"
    if close(g, 1.0) and close(k, 0.0):  return "Hardness-only"

    if m == "cgh":
        return "CGH"

    return m.upper() if m else "UNKNOWN"


def bold_best(vals_raw: Dict[str, float], vals_fmt: Dict[str, str]) -> Dict[str, str]:
    mx = np.nanmax(list(vals_raw.values()))
    out = {}
    for v, s in vals_fmt.items():
        if np.isfinite(vals_raw[v]) and vals_raw[v] >= mx - 1e-12:
            out[v] = "\\textbf{" + s + "}"
        else:
            out[v] = s
    return out


def expand_inputs(inputs: List[str]) -> List[str]:
    files = []
    for x in inputs:
        files.extend(glob.glob(x, recursive=True))
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "stl10"])
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more CSV files or globs, e.g. runs/**/robustness_summary.csv")
    ap.add_argument("--forced_dataset", action="store_true",
                    help="Force dataset name onto files that lack a dataset column (uses --dataset).")
    ap.add_argument("--seeds", default=SEEDS_DEFAULT)
    ap.add_argument("--order", default="SimCLR,Gate-only,Mild CGH,Full CGH,Hardness-only")
    args = ap.parse_args()

    seeds = set(int(s) for s in args.seeds.split(",") if s.strip())
    order = [s.strip() for s in args.order.split(",") if s.strip()]

    files = expand_inputs(args.inputs)
    if not files:
        raise SystemExit("No input files matched.")

    dfs = []
    for fp in files:
        d = pd.read_csv(fp)
        d = normalize_columns(d)
        d = ensure_gamma_kg(d)
        d = force_dataset_if_missing(d, args.dataset if args.forced_dataset else None)
        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)

    df["dataset"] = df["dataset"].astype(str).str.lower()
    df = df[df["dataset"].eq(args.dataset)].copy()
    df = df[df["seed"].isin(seeds)].copy()

    df["variant"] = df.apply(infer_variant, axis=1)
    df = df[df["variant"].isin(order)].copy()

    # include arch/epochs to avoid dropping legitimate runs
    dedup_cols = ["dataset", "variant", "seed", "pos_corrupt_p"]
    for c in ["arch", "epochs", "gamma", "k_gate"]:
        if c in df.columns:
            dedup_cols.append(c)
    df = df.drop_duplicates(subset=dedup_cols)

    cov = (df.groupby(["pos_corrupt_p", "variant"])["seed"]
             .nunique()
             .unstack("variant", fill_value=0)
             .sort_index())
    print("n_seeds coverage per p (expect 5):")
    print(cov)

    agg = (df.groupby(["pos_corrupt_p", "variant"], as_index=False)
             .agg(
                 knn_mean=("knn_acc", "mean"),
                 knn_std=("knn_acc", sample_std),
                 lin_mean=("linear_acc", "mean"),
                 lin_std=("linear_acc", sample_std),
                 n_seeds=("seed", "nunique"),
             ))

    ps = sorted([p for p in agg["pos_corrupt_p"].dropna().unique()])

    print("\n% ---- kNN accuracy (%) ----")
    for p in ps:
        row = agg[agg["pos_corrupt_p"] == p].set_index("variant")
        raw, fmt = {}, {}
        for v in order:
            if v not in row.index:
                raw[v], fmt[v] = np.nan, "NA"
            else:
                raw[v] = float(row.loc[v, "knn_mean"])
                fmt[v] = format_mean_std_pct(row.loc[v, "knn_mean"], row.loc[v, "knn_std"])
        out = bold_best(raw, fmt)
        print(f"{p:.1f} & " + " & ".join(out[v] for v in order) + r" \\")

    print("\n% ---- Linear probe accuracy (%) ----")
    for p in ps:
        row = agg[agg["pos_corrupt_p"] == p].set_index("variant")
        raw, fmt = {}, {}
        for v in order:
            if v not in row.index:
                raw[v], fmt[v] = np.nan, "NA"
            else:
                raw[v] = float(row.loc[v, "lin_mean"])
                fmt[v] = format_mean_std_pct(row.loc[v, "lin_mean"], row.loc[v, "lin_std"])
        out = bold_best(raw, fmt)
        print(f"{p:.1f} & " + " & ".join(out[v] for v in order) + r" \\")


if __name__ == "__main__":
    main()
