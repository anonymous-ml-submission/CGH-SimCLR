#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/eval_only_cifar_all_final.py

Evaluate a trained checkpoint on CIFAR-10-C / CIFAR-100-C over ALL corruptions and severities.
Designed for artifact: repo-relative imports, offline-friendly.

Outputs:
- per-corruption/severity CSV (default: cifarc_eval.csv)
- also prints mean accuracy over all corruptions/severities.

Notes:
- Uses kNN on frozen backbone features (rep='h') by default, matching your training script.
- Linear probe is optional (slow). Default: off.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

from cgh_simclr.model import SimCLRModel
from cgh_simclr.transforms import build_eval_transform
from cgh_simclr.cifarc import CIFARC
from cgh_simclr.eval_utils import extract_features, knn_predict, accuracy, train_linear_probe


CIFAR_C_CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise",
    "defocus_blur","glass_blur","motion_blur","zoom_blur",
    "snow","frost","fog",
    "brightness","contrast","elastic_transform","pixelate","jpeg_compression",
    "speckle_noise","gaussian_blur","spatter","saturate",
]


def load_ckpt(ckpt_path: str, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("Checkpoint must be a dict with key 'model' (saved by scripts/train.py).")
    args = ckpt.get("args", {}) or {}
    return ckpt, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to ckpt_epXXX.pt produced by scripts/train.py")

    ap.add_argument("--dataset", default=None, choices=["cifar10","cifar100"],
                    help="If omitted, inferred from checkpoint args.")
    ap.add_argument("--data_root", default="./data", help="Root folder that contains CIFAR train files.")
    ap.add_argument("--c_root", default="./data/CIFAR-C", help="Folder containing CIFAR-10-C/ or CIFAR-100-C/")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--arch", default=None, choices=["resnet18","resnet50"],
                    help="If omitted, inferred from checkpoint args.")
    ap.add_argument("--img_size", type=int, default=None,
                    help="If omitted, inferred from checkpoint args (32 for CIFAR).")
    ap.add_argument("--cifar_stem", action="store_true",
                    help="Force CIFAR stem. If omitted, inferred from checkpoint args when available.")

    ap.add_argument("--rep", default="h", choices=["h","p","z"], help="Feature rep for kNN.")
    ap.add_argument("--knn_k", type=int, default=200)
    ap.add_argument("--knn_temp", type=float, default=0.07)

    ap.add_argument("--do_linear", action="store_true", help="Also run linear probe for each corruption (slow).")
    ap.add_argument("--linear_epochs", type=int, default=50)
    ap.add_argument("--linear_lr", type=float, default=0.1)
    ap.add_argument("--linear_wd", type=float, default=0.0)

    ap.add_argument("--corruptions", default="all",
                    help="Comma list of corruptions, or 'all' (default).")
    ap.add_argument("--severities", default="1,2,3,4,5",
                    help="Comma list of severities (default: 1..5).")

    ap.add_argument("--out_csv", default="cifarc_eval.csv")
    ap.add_argument("--no_download", action="store_true",
                    help="Never attempt dataset downloads (recommended for reviewers).")
    args = ap.parse_args()

    ckpt, ckpt_args = load_ckpt(args.ckpt, map_location="cpu")

    dataset = (args.dataset or ckpt_args.get("dataset") or "cifar10").lower()
    if dataset not in ("cifar10","cifar100"):
        raise ValueError(f"Unsupported dataset: {dataset}")

    arch = args.arch or ckpt_args.get("arch") or "resnet18"
    img_size = args.img_size if args.img_size is not None else int(ckpt_args.get("img_size") or 32)

    # infer cifar_stem from ckpt args if present
    cifar_stem = bool(args.cifar_stem) or bool(ckpt_args.get("cifar_stem", True))

    # normalization: prefer audit stats saved in ckpt; else fallback to CIFAR defaults
    audit = ckpt.get("audit", None) or {}
    mean = audit.get("pixel_mean", None)
    std = audit.get("pixel_std", None)
    if mean is None or std is None or any(m is None for m in mean):
        # CIFAR mean/std are stable defaults; use them only if audit missing
        mean = [0.4914, 0.4822, 0.4465] if dataset == "cifar10" else [0.5071, 0.4867, 0.4408]
        std  = [0.2470, 0.2435, 0.2616] if dataset == "cifar10" else [0.2675, 0.2565, 0.2761]

    device = torch.device(args.device)

    model = SimCLRModel(arch=arch, proj_dim=128, cifar_stem=cifar_stem).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    eval_t = build_eval_transform(dataset, img_size=img_size, mean=mean, std=std)

    # build labeled CIFAR train set for kNN & linear probe training
    DS = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    train_lab = DS(root=args.data_root, train=True, download=(not args.no_download), transform=eval_t)

    train_loader = torch.utils.data.DataLoader(
        train_lab, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=(device.type=="cuda")
    )

    # Precompute train features once
    train_feats, train_y = extract_features(model, train_loader, device, rep=args.rep, normalize=True)

    if args.corruptions.strip().lower() == "all":
        corruptions = CIFAR_C_CORRUPTIONS
    else:
        corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]
        unknown = [c for c in corruptions if c not in CIFAR_C_CORRUPTIONS]
        if unknown:
            raise ValueError(f"Unknown corruption(s): {unknown}. Use --corruptions all or valid CIFAR-C names.")

    severities = [int(s) for s in args.severities.split(",") if s.strip()]
    for s in severities:
        if s < 1 or s > 5:
            raise ValueError("Severity must be in {1,2,3,4,5}")

    rows = []
    for c_type in corruptions:
        for c_sev in severities:
            test_lab = CIFARC(
                root=args.c_root,
                base=dataset,
                corruption=c_type,
                severity=c_sev,
                transform=eval_t
            )
            test_loader = torch.utils.data.DataLoader(
                test_lab, batch_size=256, shuffle=False,
                num_workers=4, pin_memory=(device.type=="cuda")
            )

            test_feats, test_y = extract_features(model, test_loader, device, rep=args.rep, normalize=True)
            pred = knn_predict(train_feats, train_y, test_feats, k=args.knn_k, temp=args.knn_temp)
            knn_acc = accuracy(pred, test_y)

            lin_acc = float("nan")
            if args.do_linear:
                lin_acc = train_linear_probe(
                    model, train_loader, test_loader, device,
                    epochs=args.linear_epochs, lr=args.linear_lr, wd=args.linear_wd
                )

            rows.append({
                "dataset": dataset,
                "arch": arch,
                "ckpt": os.path.basename(args.ckpt),
                "rep": args.rep,
                "c_type": c_type,
                "c_sev": int(c_sev),
                "knn_acc": float(knn_acc),
                "linear_acc": float(lin_acc),
            })
            print(f"[{c_type:18s} sev={c_sev}] kNN={knn_acc*100:.2f}%"
                  + (f"  LIN={lin_acc*100:.2f}%" if np.isfinite(lin_acc) else ""))

    # save
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # summary
    mean_knn = float(df["knn_acc"].mean())
    print("\n[SUMMARY] mean kNN over all corruptions/severities:", f"{mean_knn*100:.2f}%")
    print("[WROTE]", str(out_csv.resolve()))


if __name__ == "__main__":
    main()
