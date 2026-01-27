#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import json
import glob
import numpy as np
import os
from collections import defaultdict

# =============================================================================
# BASE = "<PATH>"
# =============================================================================
BASE = "<PATH>"

# change this if needed
# =============================================================================
# METHOD_GLOB = "baseline_ds=cifar10*"   # or cgh_ds=cifar10*
# =============================================================================
# =============================================================================
# METHOD_GLOB = "cgh_ds=cifar10*"   # or baseline_ds=cifar10*
# =============================================================================
METHOD_GLOB = "cgh_ds=cifar100*"   # or baseline_ds=cifar10*

P_VALUES = ["0.00", "0.20", "0.30", "0.40"]

def read_log(path):
    knn, linear = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            j = json.loads(line)
            knn.append(j["knn_acc"])
            linear.append(j["linear_acc"])
    return np.array(knn), np.array(linear)

results = {}

for p in P_VALUES:
    seed_knn = []
    seed_linear = []

    logs = glob.glob(
        f"{BASE}/{METHOD_GLOB}_p={p}_*_seed=*/cifar100c_ep200_eval.log"
    )

    print(f"\n[p={p}] found {len(logs)} logs")

    for log in sorted(logs):
        knn, linear = read_log(log)

        # sanity check: must be 75 = 15 corruptions × 5 severities
        if len(knn) != 75:
            print(f"[WARN] {log}: found {len(knn)} entries (expected 75)")
            continue

        seed_knn.append(knn.mean())
        seed_linear.append(linear.mean())

    seed_knn = np.array(seed_knn)
    seed_linear = np.array(seed_linear)

    results[p] = {
        "knn_mean": seed_knn.mean(),
        "knn_std": seed_knn.std(ddof=1),
        "linear_mean": seed_linear.mean(),
        "linear_std": seed_linear.std(ddof=1),
        "n_seeds": len(seed_knn),
    }

# ---- print table-ready output ----
print("\n=== CIFAR-100-C summary (mean over 15×5, then mean±std over seeds) ===\n")
for p in P_VALUES:
    r = results[p]
    print(
        f"p={p}: "
        f"kNN {100*r['knn_mean']:.2f} ± {100*r['knn_std']:.2f}, "
        f"Linear {100*r['linear_mean']:.2f} ± {100*r['linear_std']:.2f} "
        f"(n={r['n_seeds']})"
    )



# =============================================================================
# python summarize_cifarc_logs.py 
# 
# =============================================================================







