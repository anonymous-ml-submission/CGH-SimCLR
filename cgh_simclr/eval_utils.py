# cgh_simclr/eval_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import l2_normalize


@torch.no_grad()
def extract_features(model, loader, device, rep="h", normalize=True):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        h, p, z = model(x)

        if rep == "h":
            f = h
        elif rep == "p":
            f = p
        elif rep == "z":
            f = z
        else:
            raise ValueError(rep)

        if normalize:
            f = l2_normalize(f, dim=1)

        feats.append(f.cpu())
        labels.append(y.cpu())

    return torch.cat(feats, 0), torch.cat(labels, 0)


@torch.no_grad()
def knn_predict(train_feats, train_labels, test_feats, k=200, temp=0.07):
    train_feats_t = train_feats.t()  # (d,N)
    train_labels = train_labels.long()
    num_classes = int(train_labels.max().item()) + 1

    preds = []
    bs = 256
    for i in range(0, test_feats.size(0), bs):
        x = test_feats[i:i+bs]  # (b,d)
        sims = torch.mm(x, train_feats_t)  # (b,N)
        topk_sims, topk_idx = sims.topk(k=k, dim=1)
        topk_labels = train_labels[topk_idx]  # (b,k)
        weights = torch.exp(topk_sims / temp)

        probs = torch.zeros(x.size(0), num_classes, device=x.device, dtype=weights.dtype)
        probs.scatter_add_(1, topk_labels, weights)
        preds.append(probs.argmax(dim=1))
    return torch.cat(preds, 0)


@torch.no_grad()
def accuracy(pred, y):
    return (pred == y).float().mean().item()


class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def _infer_n_classes_from_dataset(ds, fallback_y_max=None):
    if hasattr(ds, "classes") and ds.classes is not None:
        return len(ds.classes)
    if hasattr(ds, "targets") and ds.targets is not None:
        ys = ds.targets
        ys = ys.tolist() if hasattr(ys, "tolist") else list(ys)
        return int(max(ys)) + 1
    if hasattr(ds, "labels") and ds.labels is not None:
        ys = ds.labels
        ys = ys.tolist() if hasattr(ys, "tolist") else list(ys)
        return int(max(ys)) + 1
    if fallback_y_max is not None:
        return int(fallback_y_max) + 1
    raise ValueError("Cannot infer number of classes from dataset.")


def train_linear_probe(model, train_loader, test_loader, device, epochs=50, lr=0.1, wd=0.0):
    model.eval()

    x0, y0 = next(iter(train_loader))
    with torch.no_grad():
        h0, _, _ = model(x0.to(device))
    in_dim = h0.size(1)

    n_classes = _infer_n_classes_from_dataset(train_loader.dataset, fallback_y_max=y0.max().item())

    probe = LinearProbe(in_dim, n_classes).to(device)
    opt = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for _ in range(epochs):
        probe.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                h, _, _ = model(x)
            logits = probe(h)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

    probe.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            h, _, _ = model(x)
            all_p.append(probe(h).argmax(dim=1).cpu())
            all_y.append(y.cpu())
    return accuracy(torch.cat(all_p), torch.cat(all_y))

