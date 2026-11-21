"""Deterministic dataset transformations. Each is applied identically to train, val and test."""
from __future__ import annotations

import torch


def pixel_permutation(seed: int, hw: int = 32 * 32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(hw, generator=g)


def permute_pixels(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Q2(b): one FIXED permutation of the 1024 spatial positions, shared by every image.
    RGB triplets move together, so every pixel value (and colour) is preserved exactly;
    only the spatial arrangement changes."""
    b, c, h, w = x.shape
    return x.flatten(2)[:, :, perm.to(x.device)].view(b, c, h, w)


def block_shuffle(x: torch.Tensor, block: int, seed: int, offset: int = 0) -> torch.Tensor:
    """Q2(c): cut each image into (32/block)^2 tiles and shuffle tile positions with a
    permutation that is DIFFERENT per image (deterministic from seed + image index).
    Local content inside each tile is intact; global layout is destroyed.
    `offset` keeps train/val/test permutations from coinciding."""
    b, c, h, w = x.shape
    n = h // block
    tiles = x.unfold(2, block, block).unfold(3, block, block)          # B,C,n,n,blk,blk
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(b, n * n, c, block, block)
    g = torch.Generator().manual_seed(seed + offset)
    perms = torch.argsort(torch.rand(b, n * n, generator=g), dim=1).to(x.device)
    tiles = torch.gather(tiles, 1, perms.view(b, n * n, 1, 1, 1).expand_as(tiles))
    tiles = tiles.view(b, n, n, c, block, block).permute(0, 3, 1, 4, 2, 5)
    return tiles.reshape(b, c, h, w).contiguous()


def block_shuffle_all(x: torch.Tensor, block: int, seed: int, bs: int = 5000) -> torch.Tensor:
    """Chunked block_shuffle; each chunk gets its own deterministic stream."""
    return torch.cat([block_shuffle(x[i:i + bs], block, seed, offset=1 + i) for i in range(0, len(x), bs)])


def gaussian_noise(x: torch.Tensor, sigma: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Q3(b) corruption: additive N(0, sigma^2) per pixel, clipped back to [0,1]."""
    noise = torch.randn(x.shape, generator=generator, device="cpu").to(x.device) if generator is not None \
        else torch.randn_like(x)
    return (x + sigma * noise).clamp_(0, 1)


def apply_chunked(fn, x: torch.Tensor, bs: int = 5000, **kw) -> torch.Tensor:
    return torch.cat([fn(x[i:i + bs], **kw) for i in range(0, len(x), bs)])
