# cgh_simclr/cifarc.py
import os
import numpy as np
from PIL import Image
import torch


class CIFARC(torch.utils.data.Dataset):
    """
    root contains:
      CIFAR-10-C/gaussian_noise.npy, labels.npy, ...
      CIFAR-100-C/gaussian_noise.npy, labels.npy, ...
    """
    def __init__(self, root, base, corruption, severity, transform=None):
        folder = "CIFAR-10-C" if base == "cifar10" else "CIFAR-100-C"
        cdir = os.path.join(root, folder)

        x = np.load(os.path.join(cdir, f"{corruption}.npy"))  # (50000,32,32,3)
        i0 = (severity - 1) * 10000
        i1 = severity * 10000

        y = np.load(os.path.join(cdir, "labels.npy"))
        if y.shape[0] == 50000:
            y = y[i0:i1]
        elif y.shape[0] == 10000:
            y = y
        else:
            raise ValueError(f"Unexpected labels shape: {y.shape}")

        self.x = x[i0:i1]
        self.y = y
        self.transform = transform

        if len(self.x) != len(self.y):
            raise ValueError("CIFAR-C x/y length mismatch")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = Image.fromarray(self.x[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[idx])

