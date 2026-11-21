"""All architectures. Every classifier takes images in [0,1] and normalises internally,
so the same model can be fed raw, transformed or autoencoder-reconstructed images."""
from __future__ import annotations

import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean).view(1, -1, 1, 1).clone())
        self.register_buffer("std", torch.as_tensor(std).view(1, -1, 1, 1).clone())

    def forward(self, x):
        return (x - self.mean) / self.std


# ----------------------------------------------------------------------------- MLPs
class MLP(nn.Module):
    """Flatten -> [Linear-BN-ReLU] x len(widths) -> Linear(10).
    BatchNorm is used in every model (MLP, Patch-MLP, CNN) so normalisation is not a confound."""

    def __init__(self, mean, std, widths, bn=True, in_dim=3 * 32 * 32):
        super().__init__()
        self.norm = Normalize(mean, std)
        layers, d = [], in_dim
        for w in widths:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w) if bn else nn.Identity(), nn.ReLU(inplace=True)]
            d = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(d, 10)

    def embed(self, x):
        return self.features(self.norm(x).flatten(1))

    def forward(self, x):
        return self.head(self.embed(x))


class PatchMLP(nn.Module):
    """MLP with a minimal spatial prior: the image is cut into non-overlapping p x p patches,
    ONE small MLP (shared weights) embeds every patch (locality + weight sharing), then a
    plain fully connected MLP classifies the concatenated patch embeddings.
    It is still made only of Linear layers; the only change is how pixels are grouped."""

    def __init__(self, mean, std, patch=4, token_dim=64, hidden=512):
        super().__init__()
        self.norm = Normalize(mean, std)
        self.p = patch
        n_tok = (32 // patch) ** 2
        pd = 3 * patch * patch
        self.token_mlp = nn.Sequential(nn.Linear(pd, token_dim), nn.LayerNorm(token_dim), nn.ReLU(inplace=True),
                                       nn.Linear(token_dim, token_dim), nn.LayerNorm(token_dim), nn.ReLU(inplace=True))
        self.features = nn.Sequential(nn.Linear(n_tok * token_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True))
        self.head = nn.Linear(hidden, 10)

    def embed(self, x):
        x = self.norm(x)
        b, p = x.shape[0], self.p
        t = x.unfold(2, p, p).unfold(3, p, p)                       # B,C,H/p,W/p,p,p
        t = t.permute(0, 2, 3, 1, 4, 5).reshape(b, -1, 3 * p * p)    # B,N,3p^2
        return self.features(self.token_mlp(t).flatten(1))

    def forward(self, x):
        return self.head(self.embed(x))


# ----------------------------------------------------------------------------- CNN
def _block(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CNN(nn.Module):
    """VGG-style: [c,c] pool [2c,2c] pool [4c,4c] pool -> global average pool -> Linear(10)."""

    def __init__(self, mean, std, c=64):
        super().__init__()
        self.norm = Normalize(mean, std)
        self.features = nn.Sequential(
            *_block(3, c), *_block(c, c), nn.MaxPool2d(2),
            *_block(c, 2 * c), *_block(2 * c, 2 * c), nn.MaxPool2d(2),
            *_block(2 * c, 4 * c), *_block(4 * c, 4 * c), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(4 * c, 10)

    def embed(self, x):
        return self.features(self.norm(x))

    def forward(self, x):
        return self.head(self.embed(x))


# ----------------------------------------------------------------------------- Autoencoder
class ConvAE(nn.Module):
    """Conv encoder 32->16->8->4 (128x4x4 = 2048 values) -> Linear bottleneck(d) -> mirror decoder.
    Only `d` changes between the three Q3 autoencoders."""

    def __init__(self, d, c=32):
        super().__init__()
        self.d = d
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, 2 * c, 3, 2, 1), nn.BatchNorm2d(2 * c), nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, 4 * c, 3, 2, 1), nn.BatchNorm2d(4 * c), nn.ReLU(inplace=True),
            nn.Flatten())
        flat = 4 * c * 4 * 4
        self.to_z = nn.Linear(flat, d)
        self.from_z = nn.Sequential(nn.Linear(d, flat), nn.ReLU(inplace=True))
        self.dec = nn.Sequential(
            nn.Unflatten(1, (4 * c, 4, 4)),
            nn.ConvTranspose2d(4 * c, 2 * c, 4, 2, 1), nn.BatchNorm2d(2 * c), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * c, c, 4, 2, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        return self.to_z(self.enc_conv(x))

    def decode(self, z):
        return self.dec(self.from_z(z))

    def forward(self, x):
        return self.decode(self.encode(x))


# ----------------------------------------------------------------------------- budget matching
def match_param_budget(build, target, lo=8, hi=8192, tol=0.02):
    """Binary-search an integer width so that count_params(build(width)) ~= target."""
    from common import count_params
    best = None
    # grow the upper bound geometrically first so we never instantiate a gigantic model
    w = lo
    while w < hi and count_params(build(w)) < target:
        w *= 2
    hi = min(hi, w)
    lo = max(lo, w // 2)
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count_params(build(mid))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (mid, n)
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    w, n = best
    if abs(n - target) / target > tol:
        print(f"[budget] warning: closest width {w} gives {n} params (target {target})")
    return w, n
