"""Q1 -- Why does an MLP do poorly on CIFAR10, and does it learn structure before memorising?

(a) Chosen reason (H_bias): the MLP has no spatial inductive bias (no locality, no weight
    sharing), so it must learn from 45k images what a spatial prior gives for free; it fits
    the training set but generalises poorly.
    Competing explanations tested at the SAME parameter and training budget:
      H_cap : the MLP simply lacks capacity (it underfits).
      H_opt : the shallow-wide MLP is the wrong shape / hard to optimise; depth would fix it.
    Models (all ~cfg.q1.param_budget parameters, same loop/optimiser/epochs/LR grid):
      mlp       1 hidden layer, width matched to budget
      mlp_deep  cfg.deep_mlp_depth hidden layers, width matched to budget          (tests H_opt)
      patch_mlp shared per-patch MLP + MLP head, matched to budget                 (tests H_bias)
    Diagnostics: train/val gap (H_cap predicts low train acc, small gap), a width sweep
    (H_cap predicts val acc keeps rising with width), and a fixed pixel-permutation control
    (H_bias predicts patch_mlp's advantage vanishes once spatial structure is destroyed,
    while the plain MLP is unaffected).

(b) Memorisation: a 2x2048 MLP trained without regularisation on labels where a fixed 40%
    are replaced by a random WRONG class. Each epoch we measure, on the training set,
    clean-subset accuracy, noisy-subset agreement with the noisy label (memorisation) and
    with the hidden true label (structure), plus clean validation accuracy.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from common import (load_cfg, get_data, run_or_load, predict, test_metrics, save_json, rdir,
                    count_params, set_seed)
from models import MLP, PatchMLP, match_param_budget
from transforms import pixel_permutation, permute_pixels


def builders(cfg, d):
    q = cfg["q1"]
    B = q["param_budget"]
    w_mlp, n1 = match_param_budget(lambda w: MLP(d.mean, d.std, [w]), B)
    w_deep, n2 = match_param_budget(lambda w: MLP(d.mean, d.std, [w] * q["deep_mlp_depth"]), B)
    h_patch, n3 = match_param_budget(
        lambda h: PatchMLP(d.mean, d.std, q["patch_size"], q["patch_token_dim"], h), B)
    spec = {
        "mlp": (lambda: MLP(d.mean, d.std, [w_mlp]), f"3072-{w_mlp}-10", n1),
        "mlp_deep": (lambda: MLP(d.mean, d.std, [w_deep] * q["deep_mlp_depth"]),
                     "3072-" + "-".join([str(w_deep)] * q["deep_mlp_depth"]) + "-10", n2),
        "patch_mlp": (lambda: PatchMLP(d.mean, d.std, q["patch_size"], q["patch_token_dim"], h_patch),
                      f"{q['patch_size']}x{q['patch_size']} patches: shared 48-{q['patch_token_dim']}-"
                      f"{q['patch_token_dim']} token MLP, then {(32 // q['patch_size']) ** 2 * q['patch_token_dim']}"
                      f"-{h_patch}-10", n3),
    }
    return spec


def tkw(cfg, epochs, lr, seed):
    t = cfg["train"]
    return dict(epochs=epochs, lr=lr, weight_decay=t["weight_decay"], seed=seed,
                batch_size=t["batch_size"], schedule=t["schedule"],
                warmup_epochs=t["warmup_epochs"], amp=t["amp"])


def tune_and_repeat(cfg, d, name, build, epochs, lr_grid, extra_seeds, prefix):
    """Identical tuning budget for every model: LR grid at seed 0, pick by val, re-run extra seeds."""
    grid = {}
    for lr in lr_grid:
        _, h = run_or_load(cfg, f"{prefix}{name}_lr{lr:g}_s0", build, d.xtr, d.ytr, d.xva, d.yva,
                           **tkw(cfg, epochs, lr, 0))
        grid[lr] = h["best_val_acc"]
    lr_best = max(grid, key=grid.get)
    runs = []
    for s in [0] + list(extra_seeds):
        m, h = run_or_load(cfg, f"{prefix}{name}_lr{lr_best:g}_s{s}", build, d.xtr, d.ytr, d.xva, d.yva,
                           **tkw(cfg, epochs, lr_best, s))
        tm = test_metrics(predict(m, d.xte), d.yte)
        runs.append({"seed": s, "run": f"{prefix}{name}_lr{lr_best:g}_s{s}", "best_val_acc": h["best_val_acc"],
                     "best_epoch": h["best_epoch"], "final_train_acc": h["train_acc"][-1],
                     "train_acc_at_best": h["train_acc"][h["best_epoch"] - 1], "test": tm})
    return {"lr_grid_val": {str(k): v for k, v in grid.items()}, "lr": lr_best, "runs": runs,
            "test_acc_mean": float(np.mean([r["test"]["acc"] for r in runs])),
            "test_acc_std": float(np.std([r["test"]["acc"] for r in runs])),
            "val_acc_mean": float(np.mean([r["best_val_acc"] for r in runs])),
            "train_acc_at_best_mean": float(np.mean([r["train_acc_at_best"] for r in runs]))}


def part_a(cfg):
    d = get_data(cfg)
    q = cfg["q1"]
    spec = builders(cfg, d)
    out = {"models": {}, "capacity_sweep": [], "permuted": {}}
    for name, (build, desc, n) in spec.items():
        print(f"[Q1a] {name}: {desc} ({n} params)")
        r = tune_and_repeat(cfg, d, name, build, q["epochs"], q["lr_grid"], q["extra_seeds"], "q1_")
        r.update(spec=desc, params=n)
        out["models"][name] = r

    # Diagnostic 1: width sweep of the plain MLP (params vary BY DESIGN; same LR as selected mlp)
    lr = out["models"]["mlp"]["lr"]
    for w in q["capacity_widths"]:
        m, h = run_or_load(cfg, f"q1_cap_w{w}_s0", lambda w=w: MLP(d.mean, d.std, [w]),
                           d.xtr, d.ytr, d.xva, d.yva, **tkw(cfg, q["epochs"], lr, 0))
        out["capacity_sweep"].append({"width": w, "params": count_params(m), "best_val_acc": h["best_val_acc"],
                                      "final_train_acc": h["train_acc"][-1],
                                      "train_acc_at_best": h["train_acc"][h["best_epoch"] - 1]})

    # Diagnostic 2: fixed pixel permutation (same perm for train/val/test). Re-used in Q2(b).
    perm = pixel_permutation(cfg["seed"])
    dp = d.with_x(permute_pixels(d.xtr, perm), permute_pixels(d.xva, perm), permute_pixels(d.xte, perm))
    for name in ("mlp", "patch_mlp"):
        build = spec[name][0]
        lr = out["models"][name]["lr"]   # hyper-parameters frozen from the clean selection
        runs = []
        for s in [0] + list(q["extra_seeds"]):
            m, h = run_or_load(cfg, f"q1_perm_{name}_lr{lr:g}_s{s}", build, dp.xtr, dp.ytr, dp.xva, dp.yva,
                               **tkw(cfg, q["epochs"], lr, s))
            runs.append({"seed": s, "best_val_acc": h["best_val_acc"],
                         "test": test_metrics(predict(m, dp.xte), dp.yte)})
        out["permuted"][name] = {"runs": runs,
                                 "test_acc_mean": float(np.mean([r["test"]["acc"] for r in runs])),
                                 "test_acc_std": float(np.std([r["test"]["acc"] for r in runs]))}
    save_json(out, rdir(cfg, "q1a.json"))
    return out


def part_b(cfg):
    d = get_data(cfg)
    q = cfg["q1"]["memorisation"]
    out = {}
    for rate in (q["noise_rate"], 0.0):        # noisy run + clean-label control
        rng = np.random.default_rng(cfg["seed"])
        y_true = d.ytr.cpu().numpy()
        y_noisy = y_true.copy()
        noisy = np.zeros(len(y_true), bool)
        if rate > 0:
            idx = rng.choice(len(y_true), int(rate * len(y_true)), replace=False)
            noisy[idx] = True
            y_noisy[idx] = (y_true[idx] + rng.integers(1, 10, size=len(idx))) % 10   # always a WRONG class
        yn = torch.from_numpy(y_noisy).to(d.xtr.device)
        yt = d.ytr
        nm = torch.from_numpy(noisy).to(d.xtr.device)
        fit_rows = []

        def cb(model, ep):
            pred = predict(model, d.xtr).argmax(1)
            fit = pred == yn
            fit_rows.append(fit.cpu().numpy())
            r = {"clean_subset_acc": fit[~nm].float().mean().item()}
            if nm.any():
                r["noisy_fit"] = fit[nm].float().mean().item()                 # memorisation
                r["noisy_true"] = (pred[nm] == yt[nm]).float().mean().item()    # structure despite wrong label
            return r   # the test set is never looked at during training

        name = f"q1_mem_noise{rate:g}"
        build = lambda: MLP(d.mean, d.std, q["widths"])
        kw = dict(epochs=q["epochs"], lr=q["lr"], weight_decay=q["weight_decay"], seed=0,
                  batch_size=cfg["train"]["batch_size"], schedule=q["schedule"], warmup_epochs=0,
                  amp=cfg["train"]["amp"], epoch_callback=cb)
        hp = rdir(cfg, "runs", f"{name}.json")
        fresh = not hp.exists()
        m, h = run_or_load(cfg, name, build, d.xtr, yn, d.xva, d.yva, **kw)
        if fresh:
            np.savez_compressed(rdir(cfg, f"{name}_fit.npz"), fit=np.stack(fit_rows), noisy=noisy)
        h["test_acc_best_val_ckpt"] = test_metrics(predict(m, d.xte), d.yte)["acc"]
        h["params"] = count_params(build())
        h["noise_rate"] = rate
        out[f"{rate:g}"] = h
    save_json(out, rdir(cfg, "q1b.json"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--part", default="ab")
    a = ap.parse_args()
    cfg = load_cfg(smoke=a.smoke)
    if "a" in a.part:
        part_a(cfg)
    if "b" in a.part:
        part_b(cfg)
