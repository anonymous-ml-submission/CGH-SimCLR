# cgh_simclr/transforms.py
import random
from typing import Tuple, Optional, List

import torch
from torchvision import transforms
from PIL import Image


def random_resized_crop_with_scale(x, base_transform: transforms.Compose):
    """
    Apply base_transform but intercept the RandomResizedCrop scale.
    We assume base_transform begins with RandomResizedCrop.
    Return transformed image + sampled scale (area fraction).
    """
    # Extract first op as RRC; apply manually to get sampled params
    ops = list(base_transform.transforms)
    if not isinstance(ops[0], transforms.RandomResizedCrop):
        # fallback: no scale info
        return base_transform(x), 1.0
    rrc: transforms.RandomResizedCrop = ops[0]
    i, j, h, w = rrc.get_params(x, rrc.scale, rrc.ratio)
    # area fraction as severity proxy (smaller crop => harder)
    img_area = (x.size[0] * x.size[1]) if hasattr(x, "size") else 1.0
    crop_area = float(h * w)
    scale = float(crop_area / max(1.0, img_area))  # in (0,1]
    x = transforms.functional.resized_crop(x, i, j, h, w, rrc.size, rrc.interpolation, rrc.antialias if hasattr(rrc, "antialias") else None)
    # apply remaining ops
    for op in ops[1:]:
        x = op(x)
    return x, scale

class TwoCropsTransform:
    def __init__(self, base_transform, return_severity_bucket=False, n_buckets=5):
        self.base_transform = base_transform
        self.return_severity_bucket = return_severity_bucket
        self.n_buckets = n_buckets

    def __call__(self, x):
        if self.return_severity_bucket:
            q, scale_q = random_resized_crop_with_scale(x, self.base_transform)
            k, scale_k = random_resized_crop_with_scale(x, self.base_transform)
            sev = 0.5 * (scale_q + scale_k)
            b = int((1.0 - sev) * self.n_buckets)
            b = max(0, min(self.n_buckets - 1, b))
            return (q, k), b
        else:
            q = self.base_transform(x)
            k = self.base_transform(x)
            return (q, k), 0




def build_ssl_transform(dataset_name: str, img_size: int, mean=None, std=None):
    name = dataset_name.lower()

    # your rule: ImageNet-scale uses (0.08,1.0), others use (0.2,1.0)
    crop_scale = (0.08, 1.0) if name in ("imagenet100", "imagenet", "imagenet1k") else (0.2, 1.0)

    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

    ops = [
        transforms.RandomResizedCrop(img_size, scale=crop_scale),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=max(3, img_size // 10 * 2 + 1), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ]
    if mean is not None and std is not None and all(m is not None for m in mean):
        ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


def build_eval_transform(dataset_name: str, img_size: int, mean=None, std=None):
    name = dataset_name.lower()
    ops = []

    # match your eval rule: do NOT resize CIFAR by default
    if name in ("stl10",):
        ops += [transforms.Resize(img_size), transforms.CenterCrop(img_size)]
    elif name in ("imagenet100",):
        ops += [transforms.Resize(256), transforms.CenterCrop(img_size)]

    ops.append(transforms.ToTensor())
    if mean is not None and std is not None and all(m is not None for m in mean):
        ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)

