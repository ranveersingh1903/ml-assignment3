"""Q3 -- Convolutional autoencoders: bottleneck capacity, denoising, and how much class
information the learned representation holds.

(a) Three ConvAEs identical except the Linear bottleneck d in cfg.q3.bottlenecks (8x steps).
    Selection rule (fixed in advance): lowest validation MSE.
(b) Corruption: additive Gaussian noise (sigma in cfg) -- reconstruction can plausibly help
    because the noise is independent per pixel while images are spatially smooth.
    DAE = same architecture as the selected AE; input = freshly-noised train image every
    batch, target = the clean image; no labels. Val/test noise is drawn once with a fixed seed.
    Fixed classifier = the Q2(a) clean CNN (seed 0), never fine-tuned.
(c) Frozen encoders -> bottleneck features -> linear probe (L2 logistic regression, weight
    decay chosen on val) and k-NN; compared with test reconstruction error. A randomly
    initialised encoder is included as a control for "architecture alone".
"""
from __future__ import annotations

import argparse
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (load_cfg, get_data, run_or_load, predict, test_metrics, save_json, load_json, rdir,
                    count_params, set_seed, device)
from models import ConvAE, CNN, match_param_budget
from q1 import tkw
from transforms import gaussian_noise


# ----------------------------------------------------------------------------- AE training
def batched(fn, x, bs=1000):
    with torch.no_grad():
        return torch.cat([fn(x[i:i + bs]) for i in range(0, len(x), bs)])


def mse(model, x_in, x_target):
    model.eval()
    rec = batched(model, x_in)
    return F.mse_loss(rec, x_target).item()


def psnr(m):
    return 10 * math.log10(1.0 / m)


def train_ae(cfg, name, d_bottleneck, d, noise_sigma=None, seed=0):
    ck, hp = rdir(cfg, "checkpoints", f"{name}.pt"), rdir(cfg, "runs", f"{name}.json")
    model = ConvAE(d_bottleneck).to(device())
    if ck.exists() and hp.exists():
        model.load_state_dict(torch.load(ck, map_location=device()))
        return model, load_json(hp)
    q = cfg["q3"]
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=q["lr"], weight_decay=0.0)
    bs, E = cfg["train"]["batch_size"], q["epochs"]
    spe = math.ceil(len(d.xtr) / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E * spe)
    g = torch.Generator().manual_seed(seed)
    gn = torch.Generator().manual_seed(cfg["seed"] + 7)
    xva_in = gaussian_noise(d.xva, noise_sigma, gn) if noise_sigma else d.xva   # fixed val corruption
    hist = {"epoch": [], "train_mse": [], "val_mse": []}
    best, best_v, t0 = None, 1e9, time.time()
    for ep in range(1, E + 1):
        model.train()
        perm = torch.randperm(len(d.xtr), generator=g).to(d.xtr.device)
        tot = 0.0
        for i in range(0, len(d.xtr), bs):
            xb = d.xtr[perm[i:i + bs]]
            xin = gaussian_noise(xb, noise_sigma) if noise_sigma else xb   # fresh noise every batch
            loss = F.mse_loss(model(xin), xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(xb)
        v = mse(model, xva_in, d.xva)
        hist["epoch"].append(ep)
        hist["train_mse"].append(tot / len(d.xtr))
        hist["val_mse"].append(v)
        if v < best_v:
            best_v, best = v, copy.deepcopy(model.state_dict())
        print(f"  [{name}] ep {ep:3d} train_mse {hist['train_mse'][-1]:.5f} val_mse {v:.5f} ({time.time()-t0:.0f}s)")
    model.load_state_dict(best)
    hist.update(best_val_mse=best_v, best_epoch=int(np.argmin(hist["val_mse"]) + 1),
                params=count_params(model), bottleneck=d_bottleneck, noise_sigma=noise_sigma)
    torch.save(model.state_dict(), ck)
    save_json(hist, hp)
    return model, hist


# ----------------------------------------------------------------------------- probes
def features(enc_model, x):
    enc_model.eval()
    return batched(enc_model.encode, x).float()


def linear_probe(ftr, ytr, fva, yva, fte, yte, wd_grid, epochs=60, seed=0):
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    ftr, fva, fte = (ftr - mu) / sd, (fva - mu) / sd, (fte - mu) / sd
    best = None
    for wd in wd_grid:
        set_seed(seed)
        lin = nn.Linear(ftr.shape[1], 10).to(ftr.device)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=wd)
        g = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            perm = torch.randperm(len(ftr), generator=g).to(ftr.device)
            for i in range(0, len(ftr), 512):
                idx = perm[i:i + 512]
                loss = F.cross_entropy(lin(ftr[idx]), ytr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            va = (lin(fva).argmax(1) == yva).float().mean().item()
            tr = (lin(ftr).argmax(1) == ytr).float().mean().item()
        if best is None or va > best["val_acc"]:
            with torch.no_grad():
                te = (lin(fte).argmax(1) == yte).float().mean().item()
            best = {"wd": wd, "val_acc": va, "train_acc": tr, "test_acc": te}
    return best


def knn_acc(ftr, ytr, fq, yq, k):
    ftr = F.normalize(ftr, dim=1)
    fq = F.normalize(fq, dim=1)
    correct = 0
    for i in range(0, len(fq), 1000):
        sim = fq[i:i + 1000] @ ftr.T
        nn_idx = sim.topk(k, dim=1).indices
        votes = F.one_hot(ytr[nn_idx], 10).sum(1)
        correct += (votes.argmax(1) == yq[i:i + 1000]).sum().item()
    return correct / len(fq)


# ----------------------------------------------------------------------------- main
def run(cfg):
    d = get_data(cfg)
    q = cfg["q3"]
    out = {"aes": {}}

    # (a) ---------------------------------------------------------------------
    aes = {}
    for b in q["bottlenecks"]:
        m, h = train_ae(cfg, f"q3_ae_d{b}", b, d)
        aes[b] = m
        out["aes"][str(b)] = {"params": h["params"], "best_val_mse": h["best_val_mse"],
                              "val_psnr": psnr(h["best_val_mse"]), "best_epoch": h["best_epoch"],
                              "final_train_mse": h["train_mse"][-1]}
    sel = min(q["bottlenecks"], key=lambda b: out["aes"][str(b)]["best_val_mse"])
    out["selected_bottleneck"] = sel
    test_mse = mse(aes[sel], d.xte, d.xte)
    out["selected_test_mse"], out["selected_test_psnr"] = test_mse, psnr(test_mse)
    for b in q["bottlenecks"]:
        out["aes"][str(b)]["test_mse"] = mse(aes[b], d.xte, d.xte)   # reported for (c) only

    # (b) ---------------------------------------------------------------------
    sig = q["noise_sigma"]
    dae, dh = train_ae(cfg, f"q3_dae_d{sel}_sigma{sig:g}", sel, d, noise_sigma=sig)
    out["dae"] = {"bottleneck": sel, "sigma": sig, "params": dh["params"], "best_val_mse_noisy_to_clean": dh["best_val_mse"],
                  "epochs": q["epochs"], "protocol": "fresh N(0,sigma^2) noise per batch, clipped to [0,1]; target = clean image; MSE; AdamW lr "
                  f"{q['lr']}, cosine, {q['epochs']} epochs, checkpoint = lowest noisy->clean val MSE"}
    # fixed classifier from Q2(a)
    c, _ = match_param_budget(lambda c: CNN(d.mean, d.std, c), cfg["q1"]["param_budget"])
    q2 = load_json(rdir(cfg, "q2.json"))
    lr = q2["models"]["cnn"]["lr"]
    clf, _ = run_or_load(cfg, f"q2_cnn_lr{lr:g}_s0", lambda: CNN(d.mean, d.std, c), d.xtr, d.ytr, d.xva, d.yva,
                         **tkw(cfg, cfg["q1"]["epochs"], lr, 0))
    gt = torch.Generator().manual_seed(cfg["seed"] + 13)
    xte_noisy = gaussian_noise(d.xte, sig, gt)     # one fixed draw of test noise
    conds = {
        "clean": d.xte,
        "noisy": xte_noisy,
        "ae(noisy)": batched(aes[sel], xte_noisy),
        "dae(noisy)": batched(dae, xte_noisy),
        "ae(clean)": batched(aes[sel], d.xte),
        "dae(clean)": batched(dae, d.xte),
    }
    out["classification"] = {}
    for k, x in conds.items():
        tm = test_metrics(predict(clf, x), d.yte)
        tm["mse_to_clean"] = F.mse_loss(x, d.xte).item()
        tm["psnr_to_clean"] = psnr(max(tm["mse_to_clean"], 1e-10))
        tm.pop("confusion")
        out["classification"][k] = tm
    a = out["classification"]
    gap = a["clean"]["acc"] - a["noisy"]["acc"]
    out["recovered_fraction"] = (a["dae(noisy)"]["acc"] - a["noisy"]["acc"]) / gap if gap > 0.01 else float("nan")
    np.savez_compressed(rdir(cfg, "q3_examples.npz"), **{k.replace("(", "_").replace(")", ""): v[:16].cpu().numpy()
                                                         for k, v in conds.items()})

    # (c) ---------------------------------------------------------------------
    encs = {f"AE d={b}": aes[b] for b in q["bottlenecks"]}
    encs[f"DAE d={sel}"] = dae
    set_seed(0)
    encs[f"random enc d={sel}"] = ConvAE(sel).to(device())
    out["probe"] = {}
    feats = {}
    for name, m in encs.items():
        ftr, fva, fte = features(m, d.xtr), features(m, d.xva), features(m, d.xte)
        feats[name] = (ftr, fva, fte)
        lp = linear_probe(ftr, d.ytr, fva, d.yva, fte, d.yte, q["probe_wd_grid"])
        kv = knn_acc(ftr, d.ytr, fva, d.yva, q["knn_k"])
        kt = knn_acc(ftr, d.ytr, fte, d.yte, q["knn_k"])
        rec = mse(m, d.xte, d.xte)
        out["probe"][name] = {"dim": ftr.shape[1], "linear": lp, "knn_val_acc": kv, "knn_test_acc": kt,
                              "test_mse_clean": rec, "test_psnr_clean": psnr(rec)}
        print(f"[Q3c] {name}: probe test {lp['test_acc']:.3f} knn {kt:.3f} mse {rec:.5f}")
    # save every candidate representation for Q4 (selection happens there, on val only)
    for name, (ftr, fva, fte) in feats.items():
        if name.startswith("random"):
            continue
        np.savez_compressed(rdir(cfg, "features", name.replace(" ", "_").replace("=", "") + ".npz"),
                            tr=ftr.cpu().numpy(), va=fva.cpu().numpy(), te=fte.cpu().numpy())
    save_json(out, rdir(cfg, "q3.json"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    run(load_cfg(smoke=ap.parse_args().smoke))
