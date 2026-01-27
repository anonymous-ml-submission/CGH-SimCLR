# cgh_simclr/loss_cgh.py
import torch
import torch.nn.functional as F


def info_nce_logits(z1, z2, tau=0.2):
    """
    Build 2B x 2B logits matrix for SimCLR and the positive index for each anchor.
    z1, z2 are assumed L2-normalized.
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, d)
    sim = torch.mm(z, z.t()) / float(tau)  # (2B, 2B)
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return sim, pos


def apply_bucket_normalization(w, bucket_ids, n_buckets: int, eps=1e-6):
    """
    Enforce E[w | bucket=b] = 1 inside the minibatch, then renormalize globally.
    """
    if n_buckets <= 1:
        return w
    w_out = w.clone()
    for b in range(int(n_buckets)):
        m = (bucket_ids == b)
        if m.any():
            denom = w[m].mean().clamp_min(eps)
            w_out[m] = w[m] / denom
    w_out = w_out / w_out.mean().clamp_min(eps)
    return w_out


def _parse_w_clip(w_clip):
    """
    Return (lo, hi) floats, or (None, None) if w_clip is None.
    Raise a clean TypeError/ValueError for invalid inputs (e.g., scalar tensors).
    """
    if w_clip is None:
        return None, None
    if torch.is_tensor(w_clip):
        if w_clip.ndim == 0:
            raise TypeError("w_clip must be a 2-tuple (lo, hi) or a length-2 tensor; got a 0-d tensor (scalar).")
        if w_clip.numel() != 2:
            raise ValueError(f"w_clip tensor must have 2 elements, got shape={tuple(w_clip.shape)} numel={w_clip.numel()}.")
        lo, hi = w_clip.flatten().tolist()
    else:
        if not isinstance(w_clip, (tuple, list)) or len(w_clip) != 2:
            raise TypeError(f"w_clip must be (lo, hi), got {type(w_clip).__name__}: {w_clip!r}")
        lo, hi = w_clip
    lo = float(lo)
    hi = float(hi)
    if not (lo > 0.0 and hi > 0.0 and hi >= lo):
        raise ValueError(f"w_clip must satisfy 0 < lo <= hi, got (lo={lo}, hi={hi}).")
    return lo, hi


def cgh_simclr_loss(
    z1, z2,
    tau=0.2,
    gamma=1.0,
    ep=1,
    gate_warmup=20,
    c0=0.75,
    t=0.08,
    w_clip=(0.25, 4.0),
    bucket_ids=None,
    n_buckets=1,
    k_gate=0.30,
    baseline=False,
):
    """
    CGH-SimCLR (Credibility-Gated Hardness SimCLR) — repo version aligned to your monolithic code.

    Steps (pair-level):
      - InfoNCE per-anchor loss_i (2B,)
      - Credibility cr from pair agreement cos(z1,z2) (detach), gate = warmup z-score then absolute sigmoid
      - Hardness from NEGATIVES-ONLY logsumexp(sim_neg) (detach), pair-averaged, robustly normalized via MAD->sigmoid
      - Gate MODULATES hardness weights (bounded), w = h^gamma * mod
      - Renorm -> clip -> renorm -> (optional bucket norm) -> renorm
      - Final loss = weighted average of loss_i with duplicated pair weights (2B,)

    Returns:
      (loss, stats_dict)
    """
    B = z1.size(0)
    device = z1.device

    sim, pos = info_nce_logits(z1, z2, tau=tau)  # (2B,2B), (2B,)
    logp = F.log_softmax(sim, dim=1)
    idx = torch.arange(2 * B, device=device)
    loss_i = -logp[idx, pos]  # (2B,)

    # baseline SimCLR
    if baseline:
        loss = loss_i.mean()
        stats = {
            "cr_mean": float("nan"),
            "cr_std": float("nan"),
            "gate_mean": float("nan"),
            "gate_std": float("nan"),
            "h_mean": float("nan"),
            "h_std": float("nan"),
            "w_std": 0.0,
            "w_min": 1.0,
            "w_max": 1.0,
            "w_sat_lo": 0.0,
            "w_sat_hi": 0.0,
            "k_gate": float(k_gate),

            "cr_t": torch.empty(0, device=device),
            "gate_t": torch.empty(0, device=device),
            "h_t": torch.empty(0, device=device),
            "w_t": torch.ones(B, device=device),

            "s_pos_mean": float("nan"),
            "s_pos_std": float("nan"),
            "s_negmax_mean": float("nan"),
            "s_negmax_std": float("nan"),
            "margin_mean": float("nan"),
            "margin_std": float("nan"),
        }
        return loss, stats

    lo, hi = _parse_w_clip(w_clip)

    with torch.no_grad():
        # Credibility (pair agreement)
        cos = (z1.detach() * z2.detach()).sum(dim=1).clamp(-1.0, 1.0)  # (B,)
        cr = 0.5 * (cos + 1.0)  # (B,) in [0,1]

        if int(ep) <= int(gate_warmup):
            mu = cr.mean()
            sd = cr.std(unbiased=False).clamp_min(1e-6)
            gate = torch.sigmoid((cr - mu) / (2.0 * sd))  # balanced early
        else:
            gate = torch.sigmoid((cr - float(c0)) / (float(t) + 1e-6))

        # Hardness (negatives-only)
        sim_neg = sim.clone()
        sim_neg[idx, idx] = -1e9
        sim_neg[idx, pos] = -1e9
        H = torch.logsumexp(sim_neg, dim=1)  # (2B,)
        h_raw = 0.5 * (H[:B] + H[B:])  # (B,)

        m = h_raw.median()
        mad = (h_raw - m).abs().median().clamp_min(1e-6)
        h = torch.sigmoid((h_raw - m) / (2.5 * mad))  # (B,) in (0,1)

        # Gate as modifier only (bounded)
        if float(k_gate) <= 0.0:
            mod = torch.ones_like(h)
        else:
            k = float(k_gate)
            alpha = 1.0 + 2.0 * k  # k=0.3 -> 1.6
            mod = (0.5 + gate).clamp(1e-3, 1.5) ** alpha

        w = (h ** float(gamma)) * mod  # (B,)

        # diagnostics in anchor space (2B,)
        s_pos = sim[idx, pos]
        s_neg_max = sim_neg.max(dim=1).values
        margin = s_neg_max - s_pos

        dbg = {
            "s_pos_mean": s_pos.mean().item(),
            "s_pos_std": s_pos.std(unbiased=False).item(),
            "s_negmax_mean": s_neg_max.mean().item(),
            "s_negmax_std": s_neg_max.std(unbiased=False).item(),
            "margin_mean": margin.mean().item(),
            "margin_std": margin.std(unbiased=False).item(),
        }

    # stabilize / clip / renorm
    w = w / w.mean().clamp_min(1e-6)
    sat_lo = sat_hi = 0.0
    if lo is not None and hi is not None:
        w = w.clamp(lo, hi)
        sat_lo = (w <= (lo + 1e-6)).float().mean().item()
        sat_hi = (w >= (hi - 1e-6)).float().mean().item()
        w = w / w.mean().clamp_min(1e-6)

    if bucket_ids is not None:
        # bucket_ids is pair-space (B,) in your pipeline
        w = apply_bucket_normalization(w, bucket_ids, n_buckets=int(n_buckets))
        w = w / w.mean().clamp_min(1e-6)

    w2 = torch.cat([w, w], dim=0)  # (2B,)
    loss = (w2 * loss_i).sum() / w2.sum().clamp_min(1e-6)

    stats = {
        "cr_mean": cr.mean().item(),
        "cr_std": cr.std(unbiased=False).item(),
        "gate_mean": gate.mean().item(),
        "gate_std": gate.std(unbiased=False).item(),
        "h_mean": h.mean().item(),
        "h_std": h.std(unbiased=False).item(),
        "w_std": w.std(unbiased=False).item(),
        "w_min": w.min().item(),
        "w_max": w.max().item(),
        "w_sat_lo": sat_lo,
        "w_sat_hi": sat_hi,
        "k_gate": float(k_gate),

        "cr_t": cr.detach(),
        "gate_t": gate.detach(),
        "h_t": h.detach(),
        "w_t": w.detach(),
    }
    stats.update(dbg)
    return loss, stats
