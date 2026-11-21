"""Shared infrastructure: config, seeding, the saved split, data, training and evaluation.

Every experiment script imports from here so that all classifiers are trained with
literally the same loop, optimiser, schedule and checkpoint-selection rule.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


# ----------------------------------------------------------------------------- config
def load_cfg(path: str | None = None, smoke: bool = False) -> dict:
    path = path or ROOT / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["smoke"] = smoke
    cfg["seed"] = seed_from_roll(cfg["roll_number"])
    if smoke:  # tiny everything: only checks that the pipeline runs end to end
        cfg["results_dir"] = "results_smoke"
        cfg["figures_dir"] = "figures_smoke"
        cfg["split_file"] = "splits/smoke_split.npz"
        cfg["val_size"] = 200
        cfg["q1"].update(epochs=2, extra_seeds=[1], capacity_widths=[64, 128], param_budget=200000)
        cfg["q1"]["memorisation"].update(epochs=3, widths=[256, 256])
        cfg["q2"].update(block_sizes=[16, 8])
        cfg["q3"].update(epochs=2, bottlenecks=[8, 32, 128])
        cfg["q4"].update(train_subset=400, C_grid=[0.01, 0.1, 1.0, 10.0, 100.0], cluster_k=[5, 10])
    for k in ("results_dir", "figures_dir"):
        (ROOT / cfg[k]).mkdir(parents=True, exist_ok=True)
    return cfg


def seed_from_roll(roll: str) -> int:
    """'All numeric characters of the roll number, in order' -> integer seed."""
    digits = re.sub(r"\D", "", roll)
    if not digits:
        raise ValueError("roll_number contains no digits")
    return int(digits)


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rdir(cfg, *parts) -> Path:
    p = ROOT / cfg["results_dir"]
    for x in parts:
        p = p / x
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ----------------------------------------------------------------------------- data
def _synthetic(n_train=1000, n_test=500, seed=0):
    """Smoke-test stand-in for CIFAR10: class-coloured blobs + noise (NOT for results)."""
    rng = np.random.default_rng(seed)

    def make(n):
        y = np.arange(n) % 10
        rng.shuffle(y)
        x = rng.normal(0.5, 0.15, size=(n, 32, 32, 3))
        for i, c in enumerate(y):
            cy, cx = rng.integers(8, 24, size=2)
            col = np.array([(c % 3) / 2, (c // 3 % 3) / 2, (c % 5) / 4])
            yy, xx = np.ogrid[:32, :32]
            m = ((yy - cy) ** 2 + (xx - cx) ** 2) < (4 + c) ** 2
            x[i][m] = 0.5 * x[i][m] + 0.5 * col
        return (np.clip(x, 0, 1) * 255).astype(np.uint8), y.astype(np.int64)

    xtr, ytr = make(n_train)
    xte, yte = make(n_test)
    return xtr, ytr, xte, yte


def load_cifar_raw(cfg):
    """Returns uint8 NHWC arrays (50000 train, 10000 official test)."""
    if cfg["smoke"]:
        return _synthetic(seed=cfg["seed"])
    from torchvision.datasets import CIFAR10
    root = ROOT / cfg["data_root"]
    tr = CIFAR10(root, train=True, download=True)
    te = CIFAR10(root, train=False, download=True)
    return tr.data, np.array(tr.targets, dtype=np.int64), te.data, np.array(te.targets, dtype=np.int64)


def get_split(cfg, y_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 45k/5k split seeded with the roll-number digits; saved once, reused forever."""
    path = ROOT / cfg["split_file"]
    if path.exists():
        z = np.load(path)
        assert int(z["seed"]) == cfg["seed"], "saved split was made with a different seed"
        return z["train_idx"], z["val_idx"]
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y_full))
    tr, va = train_test_split(idx, test_size=cfg["val_size"], stratify=y_full,
                              random_state=cfg["seed"] % (2**32), shuffle=True)
    tr, va = np.sort(tr), np.sort(va)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_idx=tr, val_idx=va, seed=cfg["seed"])
    h = hashlib.sha256(tr.tobytes() + va.tobytes()).hexdigest()[:16]
    print(f"[split] created {path} seed={cfg['seed']} sha256[:16]={h}")
    return tr, va


@dataclass
class Data:
    """float32 tensors in [0,1], NCHW, already on the compute device."""
    xtr: torch.Tensor
    ytr: torch.Tensor
    xva: torch.Tensor
    yva: torch.Tensor
    xte: torch.Tensor
    yte: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def with_x(self, xtr, xva, xte) -> "Data":
        d = copy.copy(self)
        d.xtr, d.xva, d.xte = xtr, xva, xte
        return d


_DATA_CACHE = {}


def get_data(cfg) -> Data:
    key = (cfg["smoke"], cfg["seed"])
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    xfull, yfull, xte, yte = load_cifar_raw(cfg)
    tr, va = get_split(cfg, yfull)
    dev = device()

    def t(x):
        return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().float().div_(255).to(dev)

    xtr, xva = t(xfull[tr]), t(xfull[va])
    d = Data(xtr, torch.from_numpy(yfull[tr]).to(dev), xva, torch.from_numpy(yfull[va]).to(dev),
             t(xte), torch.from_numpy(yte).to(dev),
             xtr.mean(dim=(0, 2, 3)), xtr.std(dim=(0, 2, 3)))   # stats from the 45k train split only
    _DATA_CACHE[key] = d
    return d


def data_hash(d: Data) -> str:
    return hashlib.sha256(d.ytr.cpu().numpy().tobytes() + d.yva.cpu().numpy().tobytes()).hexdigest()[:12]


# ----------------------------------------------------------------------------- training
def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _lr_lambda(schedule, epochs, steps_per_epoch, warmup_epochs):
    total = epochs * steps_per_epoch
    warm = warmup_epochs * steps_per_epoch

    def f(step):
        if schedule == "constant":
            return 1.0
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    return f


@torch.no_grad()
def predict(model, x, bs=1000) -> torch.Tensor:
    """Returns logits (float32) for x in [0,1]."""
    model.eval()
    out = []
    use_amp = x.is_cuda
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out.append(model(x[i:i + bs]).float())
    return torch.cat(out)


def accuracy(logits, y) -> float:
    return (logits.argmax(1) == y).float().mean().item()


def train_classifier(model, xtr, ytr, xva, yva, *, epochs, lr, weight_decay, seed,
                     batch_size=256, schedule="cosine", warmup_epochs=1, amp=True,
                     epoch_callback=None, tag=""):
    """The single training loop used by every classifier in the assignment.

    Returns (best_model, history). `best_model` is the epoch with the highest
    validation accuracy (validation is the only selection signal).
    """
    set_seed(seed)
    dev = xtr.device
    model = model.to(dev)
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    spe = math.ceil(len(xtr) / batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(schedule, epochs, spe, warmup_epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dev.type == "cuda")
    g = torch.Generator(device="cpu").manual_seed(seed)
    hist = {k: [] for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time")}
    best, best_acc = None, -1.0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=g).to(dev)
        tot_loss, tot_correct = 0.0, 0
        for i in range(0, len(xtr), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = xtr[idx], ytr[idx]
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                out = model(xb)
                loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += loss.item() * len(idx)
            tot_correct += (out.argmax(1) == yb).sum().item()
        lv = predict(model, xva)
        val_loss = F.cross_entropy(lv, yva).item()
        val_acc = accuracy(lv, yva)
        # train metrics are re-measured in eval mode (no dropout/BN noise) for clean curves
        lt = predict(model, xtr)
        hist["epoch"].append(ep)
        hist["train_loss"].append(F.cross_entropy(lt, ytr).item())
        hist["train_acc"].append(accuracy(lt, ytr))
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        hist["time"].append(time.time() - t0)
        if epoch_callback is not None:
            for k, v in epoch_callback(model, ep).items():
                hist.setdefault(k, []).append(v)
        if val_acc > best_acc:
            best_acc, best = val_acc, copy.deepcopy(model.state_dict())
        print(f"  [{tag}] ep {ep:3d} train_acc {hist['train_acc'][-1]:.3f} "
              f"val_acc {val_acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    hist["best_epoch"] = int(np.argmax(hist["val_acc"]) + 1)
    hist["best_val_acc"] = best_acc
    model.load_state_dict(best)
    return model, hist


def run_or_load(cfg, name, build_fn, xtr, ytr, xva, yva, **train_kw):
    """Caches every trained classifier (checkpoint + history) so runs are resumable and
    models can be reused across questions exactly as first trained."""
    ck = rdir(cfg, "checkpoints", f"{name}.pt")
    hp = rdir(cfg, "runs", f"{name}.json")
    model = build_fn()
    if ck.exists() and hp.exists():
        model.load_state_dict(torch.load(ck, map_location=device()))
        return model.to(device()), load_json(hp)
    model, hist = train_classifier(model, xtr, ytr, xva, yva, tag=name, **train_kw)
    hist["params"] = count_params(model)
    hist["train_kw"] = {k: v for k, v in train_kw.items() if k != "epoch_callback"}
    torch.save(model.state_dict(), ck)
    save_json(hist, hp)
    return model, hist


def test_metrics(logits: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> dict:
    """Accuracy, NLL, confidence statistics and expected calibration error."""
    p = logits.softmax(1)
    conf, pred = p.max(1)
    correct = pred == y
    bins = torch.linspace(0, 1, n_bins + 1, device=p.device)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(correct[m].float().mean().item() - conf[m].mean().item())
    return {
        "acc": correct.float().mean().item(),
        "nll": F.cross_entropy(logits, y).item(),
        "mean_conf": conf.mean().item(),
        "conf_correct": conf[correct].mean().item() if correct.any() else float("nan"),
        "conf_wrong": conf[~correct].mean().item() if (~correct).any() else float("nan"),
        "ece": ece,
        "per_class_acc": [correct[y == c].float().mean().item() for c in range(10)],
        "confusion": confusion(pred, y).tolist(),
    }


def confusion(pred, y, k=10) -> np.ndarray:
    m = np.zeros((k, k), dtype=np.int64)
    np.add.at(m, (y.cpu().numpy(), pred.cpu().numpy()), 1)
    return m
