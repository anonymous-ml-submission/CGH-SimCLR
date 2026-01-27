import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def l2_normalize(x, dim=1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)

class ResNetBackbone(nn.Module):
    def __init__(self, arch="resnet18", cifar_stem=True):
        super().__init__()
        if arch == "resnet18":
            m = models.resnet18(weights=None)
            feat_dim = 512
        elif arch == "resnet50":
            m = models.resnet50(weights=None)
            feat_dim = 2048
        else:
            raise ValueError

        if cifar_stem:
            m.conv1 = nn.Conv2d(3, m.conv1.out_channels, 3, 1, 1, bias=False)
            m.maxpool = nn.Identity()

        self.encoder = nn.Sequential(*list(m.children())[:-1])
        self.feat_dim = feat_dim

    def forward(self, x):
        return self.encoder(x).flatten(1)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x):
        return self.net(x)

class SimCLRModel(nn.Module):
    def __init__(self, arch="resnet18", proj_dim=128, cifar_stem=True):
        super().__init__()
        self.backbone = ResNetBackbone(arch, cifar_stem)
        self.projector = ProjectionHead(self.backbone.feat_dim, proj_dim)

    def forward(self, x):
        h = self.backbone(x)
        p = self.projector(h)
        z = l2_normalize(p)
        return h, p, z
