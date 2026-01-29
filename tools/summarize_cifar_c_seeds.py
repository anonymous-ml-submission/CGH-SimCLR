#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

#!/usr/bin/env python3
import pandas as pd
import numpy as np

def std_ddof1(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_seed", default="cifar10c_seed_summary.csv")
    ap.add_argument("--out_group", default="cifar10c_group_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df = df.drop_duplicates(subset=[
        "dataset","method","p","gamma","k_gate","seed","ckpt_ep",
        "c_type","c_sev"
    ])
    # -------- per-seed summary (average over 75 CIFAR-C conditions) --------
    seed = (
        df.groupby(["dataset","method","p","gamma","k_gate","seed","epochs","arch","ckpt_ep"], as_index=False)
          .agg(
              n=("c_type","count"),
              n_corr=("c_type","nunique"),
              n_sev=("c_sev","nunique"),
              knn_mean=("knn_acc","mean"),
              lin_mean=("linear_acc","mean"),         
              lin_count=("linear_acc","count"),       
          )
    )

    # Flag incomplete seeds (expected n=75, n_corr=15, n_sev=5)
    seed["complete"] = (seed["n"] == 75) & (seed["n_corr"] == 15) & (seed["n_sev"] == 5)

    seed.to_csv(args.out_seed, index=False)

    # -------- group mean±std across seeds --------
    # 
    seed_ok = seed[seed["complete"]].copy()

    group = (
        seed_ok.groupby(["dataset","method","p","gamma","k_gate","epochs","arch","ckpt_ep"], as_index=False)
              .agg(
                  n_seeds=("seed","nunique"),
                  n_rows=("n","sum"),
                  knn_mean=("knn_mean","mean"),
                  knn_std=("knn_mean", std_ddof1),
                  lin_mean=("lin_mean","mean"),          # will ignore NaNs
                  lin_std=("lin_mean", std_ddof1),
                  lin_count_total=("lin_count","sum"),
              )
              .sort_values(["dataset","p","method","gamma","k_gate"])
    )

    group.to_csv(args.out_group, index=False)

    print("Wrote:", args.out_seed)
    print("Wrote:", args.out_group)
    print(group)

if __name__ == "__main__":
    main()






