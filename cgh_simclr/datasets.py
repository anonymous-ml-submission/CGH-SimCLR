# cgh_simclr/datasets.py
import os
from typing import Optional, Callable
from torchvision import datasets

def _cifar_present(root: str, dataset: str) -> bool:
    dataset = dataset.lower()
    if dataset == "cifar10":
        return os.path.isdir(os.path.join(root, "cifar-10-batches-py"))
    if dataset == "cifar100":
        return os.path.isdir(os.path.join(root, "cifar-100-python"))
    return False

def _stl10_present(root: str) -> bool:
    # torchvision STL10 uses "stl10_binary" under root
    return os.path.isdir(os.path.join(root, "stl10_binary"))

def get_cifar(
    dataset: str,
    root: str,
    train: bool,
    allow_download: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    dataset = dataset.lower()
    if dataset not in ("cifar10", "cifar100"):
        raise ValueError("dataset must be cifar10 or cifar100")

    present = _cifar_present(root, dataset)
    if (not present) and (not allow_download):
        need = "cifar-10-batches-py" if dataset == "cifar10" else "cifar-100-python"
        raise FileNotFoundError(
            f"{dataset} not found under data_root='{root}'. "
            f"Expected folder: '{os.path.join(root, need)}'. "
            f"Either place the dataset there, or rerun without --no_download."
        )

    download = bool(allow_download) and (not present)
    DS = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    return DS(
        root=root,
        train=train,
        download=download,
        transform=transform,
        target_transform=target_transform,
    )

def get_stl10(
    root: str,
    split: str,
    allow_download: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    present = _stl10_present(root)
    if (not present) and (not allow_download):
        need = os.path.join(root, "stl10_binary")
        raise FileNotFoundError(
            f"stl10 not found under data_root='{root}'. "
            f"Expected folder: '{need}'. "
            f"Either place the dataset there, or rerun without --no_download."
        )

    download = bool(allow_download) and (not present)
    return datasets.STL10(
        root=root,
        split=split,
        download=download,
        transform=transform,
        target_transform=target_transform,
    )
