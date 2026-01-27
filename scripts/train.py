#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train.py

CGH-SimCLR (Credibility-Gated Hardness SimCLR) + Deep Dataset Audit + kNN + Linear Probe
Repo version MATCHED to your original (monolithic) code that generated results.

Key behaviors preserved:
- pair-level weights w = (h^gamma) * gate_modifier
- hardness from negatives-only logsumexp(sim_neg) (detach) then MAD-sigmoid normalization
- warmup gate (batch z-scored) then absolute gate sigmoid((cr-c0)/t)
- clip weights + renormalize to mean 1; optional bucket-normalization (exogenous)
- optional positive-pair corruption by in-batch swapping
- optional VICReg-style variance/cov regularizers on pre-norm projector outputs
- outputs: args.json, audit.json, eval.json, ckpts, robustness_summary.csv, dynamics.csv, w_hist.csv
"""

import os, argparse, time, json, random, statistics, csv
from dataclasses import dataclass
from collections import Counter
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from cgh_simclr.model import SimCLRModel
from cgh_simclr.transforms import TwoCropsTransform, build_ssl_transform, build_eval_transform
from cgh_simclr.loss_cgh import cgh_simclr_loss
from cgh_simclr.cifarc import CIFARC
from cgh_simclr.eval_utils import extract_features, knn_predict, accuracy, train_linear_probe
from cgh_simclr.datasets import get_cifar, get_stl10


# -------------------------
# Reproducibility
# -------------------------
def seed_all(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


# -------------------------
# Deep dataset audit (counts + mean/std)
# -------------------------
@dataclass
class AuditReport:
    n: int
    n_classes: int
    class_counts: dict
    class_imbalance_ratio: float
    num_corrupt: int
    pixel_mean: list
    pixel_std: list


def audit_dataset(ds, max_items=20000):
    from torchvision import transforms
    tt = transforms.ToTensor()

    n_total = len(ds)
    n_scan = min(n_total, max_items)

    class_counts = Counter()
    corrupt = 0

    ch_sum = np.zeros(3, dtype=np.float64)
    ch_sumsq = np.zeros(3, dtype=np.float64)
    n_pix = 0

    for idx in range(n_scan):
        try:
            item = ds[idx]
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                img, y = item[0], item[1]
                if isinstance(y, (int, np.integer)):
                    class_counts[int(y)] += 1
            else:
                img = item

            t = tt(img)  # (C,H,W) in [0,1]
            if t.size(0) == 1:
                t = t.repeat(3, 1, 1)
            c, h, w = t.shape
            t2 = t.view(c, -1).double().cpu().numpy()
            ch_sum += t2.sum(axis=1)
            ch_sumsq += (t2 ** 2).sum(axis=1)
            n_pix += t2.shape[1]
        except Exception:
            corrupt += 1

    n_classes = len(class_counts) if len(class_counts) else 0
    if n_classes > 0:
        mx = max(class_counts.values())
        mn = min(class_counts.values())
        imbalance_ratio = float(mx) / float(mn) if mn > 0 else float("inf")
    else:
        imbalance_ratio = float("nan")

    if n_pix > 0:
        mean = (ch_sum / n_pix).tolist()
        var = (ch_sumsq / n_pix - (ch_sum / n_pix) ** 2)
        var = np.clip(var, 0.0, None)
        std = np.sqrt(var).tolist()
    else:
        mean, std = [None] * 3, [None] * 3

    return AuditReport(
        n=n_scan,
        n_classes=n_classes,
        class_counts=dict(class_counts),
        class_imbalance_ratio=imbalance_ratio,
        num_corrupt=corrupt,
        pixel_mean=mean,
        pixel_std=std,
    )


# -------------------------
# Dataset builders
# -------------------------
def build_raw_dataset(name, root, train=True, allow_download=True):
    name = name.lower()
    if name in ("cifar10", "cifar100"):
        return get_cifar(name, root=root, train=train, allow_download=allow_download)
    if name == "stl10":
        split = "train" if train else "test"
        return get_stl10(root=root, split=split, allow_download=allow_download)
    if name == "imagenet100":
        split = "train" if train else "val"
        return ImageFolder(root=os.path.join(root, "imagenet100", split))
    raise ValueError("dataset must be cifar10/cifar100/stl10/imagenet100")


def build_ssl_dataset(name, root, train=True, img_size=32, mean=None, std=None,
                      allow_download=True, bucket_norm=False, n_buckets=5):
    name = name.lower()
    if name in ("cifar10", "cifar100"):
        base = get_cifar(name, root=root, train=train, allow_download=allow_download)
    elif name == "stl10":
        base = get_stl10(root=root, split="unlabeled", allow_download=allow_download)
    elif name == "imagenet100":
        base = ImageFolder(root=os.path.join(root, "imagenet100", "train" if train else "val"))
    else:
        raise ValueError("dataset must be cifar10/cifar100/stl10/imagenet100")

    ssl_t = build_ssl_transform(name, img_size, mean=mean, std=std)
    base.transform = TwoCropsTransform(
        ssl_t,
        return_severity_bucket=bool(bucket_norm),
        n_buckets=int(n_buckets),
    )
    return base


# -------------------------
# VICReg regularizers (as in your original)
# -------------------------
def vicreg_variance_loss(z, nu=1.0, eps=1e-4):
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(float(nu) - std) ** 2)


def vicreg_covariance_loss(z):
    N, d = z.shape
    if N <= 1:
        return z.new_tensor(0.0)
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (N - 1)
    offdiag = cov.flatten()[1:].view(d - 1, d + 1)[:, :-1].flatten()
    return (offdiag ** 2).mean()


# -------------------------
# CSV logging (same filenames as your original)
# -------------------------
def append_result_row(out_root, args, knn_acc, lin_acc):
    path = os.path.join(out_root, "robustness_summary.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)

    row = {
        "time": int(time.time()),
        "dataset": args.dataset,
        "arch": args.arch,
        "epochs": args.epochs,
        "seed": args.seed,
        "pos_corrupt_p": float(getattr(args, "pos_corrupt_p", 0.0)),
        "method": "baseline" if args.baseline else "cgh",
        "knn_acc": float(knn_acc),
        "linear_acc": float(lin_acc),
        "gamma": float(getattr(args, "gamma", float("nan"))),
        "k_gate": float(getattr(args, "k_gate", float("nan"))),
        "wmin": float(getattr(args, "wmin", float("nan"))),
        "wmax": float(getattr(args, "wmax", float("nan"))),
        "lambda_var": float(getattr(args, "lambda_var", float("nan"))),
        "lambda_cov": float(getattr(args, "lambda_cov", float("nan"))),
    }

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def append_dynamics_row(out_root, args, ep,
                        ess_list, corr_wh_list, corr_wcr_list,
                        wmin_list, wmax_list, wstd_list, gateon_list):
    path = os.path.join(out_root, "dynamics.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)

    def mean_or_nan(x):
        return float(sum(x) / len(x)) if len(x) else float("nan")

    row = {
        "dataset": args.dataset,
        "arch": args.arch,
        "epochs": args.epochs,
        "seed": args.seed,
        "pos_corrupt_p": float(getattr(args, "pos_corrupt_p", 0.0)),
        "method": "baseline" if args.baseline else "cgh",
        "epoch": int(ep),
        "ESS_mean": mean_or_nan(ess_list),
        "corr_w_h_mean": mean_or_nan(corr_wh_list),
        "corr_w_cr_mean": mean_or_nan(corr_wcr_list),
        "w_min_mean": mean_or_nan(wmin_list),
        "w_max_mean": mean_or_nan(wmax_list),
        "w_std_mean": mean_or_nan(wstd_list),
        "gate_on_frac_mean": mean_or_nan(gateon_list),
    }

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def append_w_hist(out_root, args, ep, w_samples):
    if not w_samples:
        return
    w_all = torch.cat(w_samples, dim=0).numpy()
    bins = np.linspace(args.wmin, args.wmax, 41)  # 40 bins
    hist, _ = np.histogram(w_all, bins=bins, density=False)
    hist = hist / (hist.sum() + 1e-12)

    path = os.path.join(out_root, "w_hist.csv")
    write_header = not os.path.exists(path)

    fieldnames = [
        "dataset", "arch", "epochs", "seed", "pos_corrupt_p", "method", "epoch",
        "gamma", "k_gate", "wmin", "wmax", "c0", "t", "gate_warmup"
    ] + [f"bin{i:02d}" for i in range(len(hist))]

    row = {
        "dataset": args.dataset,
        "arch": args.arch,
        "epochs": args.epochs,
        "seed": args.seed,
        "pos_corrupt_p": float(getattr(args, "pos_corrupt_p", 0.0)),
        "method": "baseline" if args.baseline else "cgh",
        "epoch": int(ep),
        "gamma": float(args.gamma),
        "k_gate": float(args.k_gate),
        "wmin": float(args.wmin),
        "wmax": float(args.wmax),
        "c0": float(args.c0),
        "t": float(args.t),
        "gate_warmup": int(args.gate_warmup),
    }
    for i, v in enumerate(hist):
        row[f"bin{i:02d}"] = float(v)

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


# -------------------------
# One full run (one seed)
# -------------------------
def main_single_run(args):
    seed_all(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    use_cuda = (device.type == "cuda")

    # default img sizes
    if args.img_size is None:
        if args.dataset == "stl10":
            img_size = 96
        elif args.dataset == "imagenet100":
            img_size = 224
        else:
            img_size = 32
    else:
        img_size = int(args.img_size)

    os.makedirs(args.data_root, exist_ok=True)

    method = "baseline" if args.baseline else "cgh"
    tag = (
        f"{method}"
        f"_ds={args.dataset}"
        f"_arch={args.arch}"
        f"_ep={args.epochs}"
        f"_tau={args.tau}"
        f"_img={img_size}"
        f"_p={args.pos_corrupt_p:.2f}"
        f"_gam={args.gamma:.2f}"
        f"_kg={args.k_gate:.2f}"
        f"_w={args.wmin:.2f}-{args.wmax:.2f}"
        f"_bv={args.bucket_norm}_nb={args.n_buckets}"
        f"_lvar={args.lambda_var:.2f}_lcov={args.lambda_cov:.2f}"
        f"_seed={args.seed}"
        f"_{int(time.time())}"
    )
    out_dir = os.path.join(args.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    allow_download = (not args.no_download)

    # ---- audit normalization stats
    if args.dataset == "imagenet100":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        report = None
    else:
        raw_train = build_raw_dataset(args.dataset, args.data_root, train=True, allow_download=allow_download)
        report = audit_dataset(raw_train, max_items=args.audit_max)
        with open(os.path.join(out_dir, "audit.json"), "w") as f:
            json.dump(report.__dict__, f, indent=2)
        print("[AUDIT]", json.dumps(report.__dict__, indent=2))
        mean, std = report.pixel_mean, report.pixel_std

    # ---- SSL dataset/loader
    ssl_train = build_ssl_dataset(
        args.dataset, args.data_root, train=True, img_size=img_size,
        mean=mean, std=std, allow_download=allow_download,
        bucket_norm=args.bucket_norm, n_buckets=args.n_buckets
    )
    ssl_loader = DataLoader(
        ssl_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        drop_last=True
    )

    if args.bucket_norm:
        from collections import Counter
        cnt = Counter()
        for ((_, _), b), _ in islice(ssl_loader, int(args.bucket_audit_batches)):
            cnt.update([int(x) for x in b])
        print("bucket_counts(sample):", dict(sorted(cnt.items())))

    # ---- Model
    cifar_stem = (args.dataset in ("cifar10", "cifar100")) and (img_size <= 64)
    if args.cifar_stem:
        cifar_stem = True

    model = SimCLRModel(arch=args.arch, proj_dim=128, cifar_stem=cifar_stem).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ---- Pretrain
    model.train()
    global_step = 0

    for ep in range(1, args.epochs + 1):
        first_batch = True
        losses, cr_means, w_stds, pos_coses = [], [], [], []

        ess_list, corr_wh_list, corr_wcr_list = [], [], []
        wmin_list, wmax_list, wstd_list, gateon_list = [], [], [], []
        w_samples = []

        for bi, (((x1, x2), bucket_id), _) in enumerate(ssl_loader):
            x1 = x1.to(device, non_blocking=use_cuda)
            x2 = x2.to(device, non_blocking=use_cuda)
            bucket_id = bucket_id.to(device, non_blocking=use_cuda)

            # corrupt positives
            if args.pos_corrupt_p > 0:
                B = x1.size(0)
                m = int(round(args.pos_corrupt_p * B))
                if first_batch:
                    print(f"[pos_corrupt] target={args.pos_corrupt_p:.2f} applied={m/B:.3f}")
                if m > 0:
                    idx = torch.randperm(B, device=device)[:m]
                    perm = torch.randperm(B, device=device)
                    bad = (perm[idx] == idx)
                    if bad.any():
                        perm[idx[bad]] = (perm[idx[bad]] + 1) % B
                    x2 = x2.clone()
                    x2[idx] = x2[perm[idx]]

            _, p1, z1 = model(x1)
            _, p2, z2 = model(x2)

            if args.debug_every and (global_step % args.debug_every == 0):
                with torch.no_grad():
                    Z_dbg = torch.cat([z1, z2], dim=0)
                    S = Z_dbg @ Z_dbg.t()
                    off = S[~torch.eye(S.size(0), dtype=torch.bool, device=S.device)]
                    print("offdiag_sim_mean:", off.mean().item(),
                          "offdiag_sim_std:", off.std(unbiased=False).item())

            with torch.no_grad():
                pos_cos = (z1 * z2).sum(dim=1).mean().item()
            pos_coses.append(pos_cos)

            loss_cgh, stats = cgh_simclr_loss(
                z1, z2,
                tau=args.tau,
                gamma=args.gamma,
                ep=ep,
                gate_warmup=args.gate_warmup,
                c0=args.c0,
                t=args.t,
                w_clip=(args.wmin, args.wmax),
                bucket_ids=bucket_id if args.bucket_norm else None,
                n_buckets=args.n_buckets if args.bucket_norm else 1,
                k_gate=args.k_gate,
                baseline=args.baseline,
            )

            if (not args.baseline) and ("w_t" in stats) and (stats["w_t"].numel() > 0):
                with torch.no_grad():
                    w = stats["w_t"]
                    h = stats["h_t"]
                    cr = stats["cr_t"]
                    gate = stats["gate_t"]

                    if (bi % 5) == 0:
                        w_samples.append(w.detach().cpu())

                    ess = (w.sum() ** 2) / (w.pow(2).sum().clamp_min(1e-12))

                    def corr(a, b):
                        a = a - a.mean()
                        b = b - b.mean()
                        return (a*b).mean() / (a.std(unbiased=False)*b.std(unbiased=False) + 1e-12)

                    ess_list.append(float(ess.item()))
                    corr_wh_list.append(float(corr(w, h).item()))
                    corr_wcr_list.append(float(corr(w, cr).item()))
                    wmin_list.append(float(w.min().item()))
                    wmax_list.append(float(w.max().item()))
                    wstd_list.append(float(w.std(unbiased=False).item()))
                    gateon_list.append(float((gate > 0.5).float().mean().item()))

            # VICReg regs on pre-norm projector outputs
            P = torch.cat([p1, p2], dim=0)
            var_loss = vicreg_variance_loss(P, nu=args.nu)
            cov_loss = vicreg_covariance_loss(P)

            opt.zero_grad(set_to_none=True)
            loss = loss_cgh + args.lambda_var * var_loss + args.lambda_cov * cov_loss

            if first_batch:
                first_batch = False
                print("z1.requires_grad", z1.requires_grad, "loss.requires_grad", loss.requires_grad)

            loss.backward()

            if args.debug_every and (global_step % args.debug_every == 0):
                print("finite_loss:", torch.isfinite(loss).item())

                if (not args.baseline) and ("w_t" in stats) and (stats["w_t"].numel() > 0):
                    with torch.no_grad():
                        cr = stats["cr_t"]; hard = stats["h_t"]; gate = stats["gate_t"]; w = stats["w_t"]
                        ess = (w.sum() ** 2) / (w.pow(2).sum().clamp_min(1e-12))
                        frac_gate_on = (gate > 0.5).float().mean().item()
                        frac_w_small = (w < 0.5).float().mean().item()
                        frac_w_large = (w > 2.0).float().mean().item()

                        def corr(a, b):
                            a = a - a.mean(); b = b - b.mean()
                            return (a*b).mean() / (a.std(unbiased=False)*b.std(unbiased=False) + 1e-12)

                        print(f"ESS={ess.item():.1f}/{w.numel()}  gate>0.5={frac_gate_on:.2f}  "
                              f"w<0.5={frac_w_small:.2f}  w>2={frac_w_large:.2f}")
                        print(f"corr(w,Cr)={corr(w,cr).item():.3f}  corr(w,h)={corr(w,hard).item():.3f}  "
                              f"corr(Cr,h)={corr(cr,hard).item():.3f}")
                        print("w_min_check", w.min().item(), "w_max_check", w.max().item())
                        if "s_pos_mean" in stats:
                            print("s_pos(mean,std):", stats["s_pos_mean"], stats["s_pos_std"],
                                  "s_negmax(mean,std):", stats["s_negmax_mean"], stats["s_negmax_std"],
                                  "margin(mean,std):", stats["margin_mean"], stats["margin_std"])

                # projector grad check
                last_linear = None
                for m in reversed(list(model.projector.net.modules())):
                    if isinstance(m, nn.Linear):
                        last_linear = m
                        break
                if last_linear is None or last_linear.weight.grad is None:
                    print("grad: None")
                else:
                    g = last_linear.weight.grad
                    print("grad_mean_abs:", g.abs().mean().item(),
                          "grad_norm:", g.norm().item(),
                          "grad_max_abs:", g.abs().max().item())

            opt.step()
            global_step += 1

            losses.append(loss.item())
            cr_means.append(float(stats.get("cr_mean", float("nan"))))
            w_stds.append(float(stats.get("w_std", 0.0)))

        sched.step()
        print(f"[PRETRAIN] ep={ep:03d} loss={statistics.mean(losses):.4f} "
              f"pos_cos={statistics.mean(pos_coses):.3f} "
              f"Cr={statistics.mean(cr_means):.3f} w_std={statistics.mean(w_stds):.3f} "
              f"lr={sched.get_last_lr()[0]:.5f}")

        if (not args.baseline):
            append_w_hist(args.out_root, args, ep, w_samples)

        if ep % 50 == 0 or ep == args.epochs:
            ckpt = {"model": model.state_dict(), "args": vars(args), "audit": (report.__dict__ if report is not None else None)}
            torch.save(ckpt, os.path.join(out_dir, f"ckpt_ep{ep:03d}.pt"))

        append_dynamics_row(args.out_root, args, ep, ess_list, corr_wh_list, corr_wcr_list,
                            wmin_list, wmax_list, wstd_list, gateon_list)

    # ---- Evaluation
    eval_t = build_eval_transform(args.dataset, img_size, mean=mean, std=std)

    if args.dataset == "stl10":
        from torchvision import datasets
        train_lab = datasets.STL10(root=args.data_root, split="train", download=allow_download, transform=eval_t)
        test_lab = datasets.STL10(root=args.data_root, split="test", download=allow_download, transform=eval_t)
    elif args.dataset == "imagenet100":
        train_lab = ImageFolder(root=os.path.join(args.data_root, "imagenet100", "train"), transform=eval_t)
        test_lab = ImageFolder(root=os.path.join(args.data_root, "imagenet100", "val"), transform=eval_t)
    else:
        from torchvision import datasets
        DS = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
        train_lab = DS(root=args.data_root, train=True, download=allow_download, transform=eval_t)
        if args.eval_c:
            test_lab = CIFARC(
                root=args.c_root,
                base=args.dataset,
                corruption=args.c_type,
                severity=args.c_sev,
                transform=eval_t,
            )
        else:
            test_lab = DS(root=args.data_root, train=False, download=allow_download, transform=eval_t)

    train_loader = DataLoader(train_lab, batch_size=256, shuffle=False, num_workers=args.num_workers, pin_memory=use_cuda)
    test_loader = DataLoader(test_lab, batch_size=256, shuffle=False, num_workers=args.num_workers, pin_memory=use_cuda)

    # kNN on backbone features (standard SimCLR)
    train_feats, train_y = extract_features(model, train_loader, device, rep="h", normalize=True)
    test_feats, test_y = extract_features(model, test_loader, device, rep="h", normalize=True)

    print("label range train:", int(train_y.min()), int(train_y.max()), "unique:", len(train_y.unique()))
    print("label range test :", int(test_y.min()), int(test_y.max()), "unique:", len(test_y.unique()))

    pred_knn = knn_predict(train_feats, train_y, test_feats, k=200, temp=0.07)
    knn_acc = accuracy(pred_knn, test_y)
    print(f"[EVAL] kNN acc = {knn_acc*100:.2f}%")

    lin_acc = train_linear_probe(model, train_loader, test_loader, device, epochs=50, lr=0.1, wd=0.0)
    print(f"[EVAL] linear probe acc = {lin_acc*100:.2f}%")

    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump({"knn_acc": knn_acc, "linear_acc": lin_acc}, f, indent=2)

    print("[DONE] outputs at:", out_dir)
    return knn_acc, lin_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "stl10", "imagenet100"])
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_root", type=str, default="runs")

    ap.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--cifar_stem", action="store_true")

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=0.2)

    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--c0", type=float, default=0.75)
    ap.add_argument("--t", type=float, default=0.08)
    ap.add_argument("--wmin", type=float, default=0.25)
    ap.add_argument("--wmax", type=float, default=4.0)
    ap.add_argument("--k_gate", type=float, default=0.30)

    ap.add_argument("--bucket_norm", action="store_true")
    ap.add_argument("--n_buckets", type=int, default=5)
    ap.add_argument("--bucket_audit_batches", type=int, default=20)

    ap.add_argument("--audit_max", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_download", action="store_true")

    ap.add_argument("--debug_every", type=int, default=0)

    ap.add_argument("--lambda_var", type=float, default=0.0)
    ap.add_argument("--nu", type=float, default=0.5)
    ap.add_argument("--lambda_cov", type=float, default=0.0)

    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--gate_warmup", type=int, default=20)

    ap.add_argument("--pos_corrupt_p", type=float, default=0.0)

    ap.add_argument("--eval_c", action="store_true")
    ap.add_argument("--c_root", type=str, default="./data/CIFAR-C")
    ap.add_argument("--c_type", type=str, default="gaussian_noise")
    ap.add_argument("--c_sev", type=int, default=1, choices=[1, 2, 3, 4, 5])

    args = ap.parse_args()
    print(f"[LOSS] method={'SimCLR' if args.baseline else 'CGH-SimCLR'} | "
          f"lambda_var={args.lambda_var} lambda_cov={args.lambda_cov}")
    # seeds as in your original "final" block
    seeds = [0, 1, 2, 3, 4]

    knn_scores, lin_scores = [], []
    for s in seeds:
        print(f"\n========== SEED {s} ==========")
        args.seed = int(s)
        knn_acc, lin_acc = main_single_run(args)
        append_result_row(args.out_root, args, knn_acc, lin_acc)
        knn_scores.append(knn_acc)
        lin_scores.append(lin_acc)

    knn_scores = np.array(knn_scores, dtype=float)
    lin_scores = np.array(lin_scores, dtype=float)

    print("\n========== FINAL (5 seeds) ==========")
    print(f"kNN accuracy    : {knn_scores.mean()*100:.2f} ± {knn_scores.std(ddof=1)*100:.2f}")
    print(f"Linear accuracy : {lin_scores.mean()*100:.2f} ± {lin_scores.std(ddof=1)*100:.2f}")


if __name__ == "__main__":
    main()
