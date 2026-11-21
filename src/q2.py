"""Q2 -- CNN vs MLP under a fair comparison, a spatial-structure ablation, and a test of
what the CNN actually relies on.

Fairness criterion (declared in configs/default.yaml and PREDICTIONS.md before running):
  matched trainable parameters (+-2%), identical data/preprocessing (no augmentation),
  optimiser, schedule, epochs, LR grid (same tuning budget), seeds and val-based checkpoint.
  FLOPs are NOT matched -- weight sharing makes the CNN reuse each parameter many times;
  that is exactly the property under study, and we report FLOPs so the reader can see it.

(a) Proposed reason for the gap: the CNN's locality + weight sharing (a spatial prior)
    lets it reuse local feature detectors across positions, so it generalises from fewer
    examples.
(b) Fixed pixel permutation (same for every image, train and test). Prediction: MLP
    unchanged (its function class is permutation-equivariant), CNN collapses towards or
    below the MLP. If the CNN kept its lead, reason (a) would be refuted.
(c) Two explanations for the CNN's behaviour:
      E-local : it classifies from local texture/colour statistics (a bag of local features)
      E-global: it needs the global spatial arrangement of object parts (shape/layout)
    Experiment: per-image random block shuffles at 16/8/4 px. The SAME transformation
    (same operator, same block size, same seeded generator) is applied to train, val and
    test; only the per-image random draw differs, as for any random transformation. E-local predicts accuracy and confidence survive 16 and 8 px shuffles and
    fall only when blocks become smaller than typical parts; E-global predicts a large drop
    already at 16 px (quadrant shuffle). Errors (per-class, confusions) and confidence
    (mean confidence on correct/wrong, ECE) are analysed per condition.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from common import load_cfg, get_data, run_or_load, predict, test_metrics, save_json, rdir, count_params
from models import CNN, match_param_budget
from q1 import builders, tune_and_repeat, tkw
from transforms import pixel_permutation, permute_pixels, block_shuffle_all


def flops(model, x):
    """Multiply-accumulates per image for Conv2d/Linear layers (forward pass)."""
    total = [0]

    def hook(m, i, o):
        if isinstance(m, torch.nn.Conv2d):
            total[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, torch.nn.Linear):
            total[0] += o.numel() // o.shape[0] * m.in_features
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(x[:2].float())
    for h in hs:
        h.remove()
    return total[0]


def run(cfg):
    d = get_data(cfg)
    q1 = cfg["q1"]
    spec = builders(cfg, d)
    c, n_cnn = match_param_budget(lambda c: CNN(d.mean, d.std, c), q1["param_budget"])
    build_cnn = lambda: CNN(d.mean, d.std, c)
    cnn_desc = f"conv3x3 [{c},{c}]-pool-[{2*c},{2*c}]-pool-[{4*c},{4*c}]-pool-GAP-{4*c}-10 (BN+ReLU)"
    out = {"models": {}, "perm": {}, "blocks": {}}

    # ---------------------------------------------------------------- (a) clean
    mlp = tune_and_repeat(cfg, d, "mlp", spec["mlp"][0], q1["epochs"], q1["lr_grid"], q1["extra_seeds"], "q1_")
    mlp.update(spec=spec["mlp"][1], params=spec["mlp"][2], reused_from="Q1(a) (runs q1_mlp_*)",
               macs=flops(spec["mlp"][0]().to(d.xtr.device), d.xtr))
    cnn = tune_and_repeat(cfg, d, "cnn", build_cnn, q1["epochs"], q1["lr_grid"], q1["extra_seeds"], "q2_")
    cnn.update(spec=cnn_desc, params=n_cnn, macs=flops(build_cnn().to(d.xtr.device), d.xtr))
    out["models"] = {"mlp": mlp, "cnn": cnn}

    # ---------------------------------------------------------------- (b) permuted pixels
    perm = pixel_permutation(cfg["seed"])          # identical permutation to Q1 diagnostic
    dp = d.with_x(permute_pixels(d.xtr, perm), permute_pixels(d.xva, perm), permute_pixels(d.xte, perm))
    conf_store = {}
    for name, build, prefix, lr in (("mlp", spec["mlp"][0], "q1_perm_mlp", mlp["lr"]),
                                    ("cnn", build_cnn, "q2_perm_cnn", cnn["lr"])):
        runs = []
        for s in [0] + list(q1["extra_seeds"]):
            m, h = run_or_load(cfg, f"{prefix}_lr{lr:g}_s{s}", build, dp.xtr, dp.ytr, dp.xva, dp.yva,
                               **tkw(cfg, q1["epochs"], lr, s))
            lg = predict(m, dp.xte)
            runs.append({"seed": s, "best_val_acc": h["best_val_acc"], "test": test_metrics(lg, dp.yte)})
            if s == 0:
                conf_store[f"perm_{name}"] = lg.softmax(1).cpu().numpy()
        out["perm"][name] = {"runs": runs, "test_acc_mean": float(np.mean([r["test"]["acc"] for r in runs])),
                             "test_acc_std": float(np.std([r["test"]["acc"] for r in runs])),
                             "reused_from": "Q1(a) permutation control" if name == "mlp" else None}

    # clean seed-0 probabilities for confidence plots
    for name, build, prefix, lr in (("mlp", spec["mlp"][0], "q1_mlp", mlp["lr"]), ("cnn", build_cnn, "q2_cnn", cnn["lr"])):
        m, _ = run_or_load(cfg, f"{prefix}_lr{lr:g}_s0", build, d.xtr, d.ytr, d.xva, d.yva, **tkw(cfg, q1["epochs"], lr, 0))
        conf_store[f"clean_{name}"] = predict(m, d.xte).softmax(1).cpu().numpy()

    # ---------------------------------------------------------------- (c) block shuffles
    for b in cfg["q2"]["block_sizes"]:
        sd = cfg["seed"] * 10 + b
        db = d.with_x(block_shuffle_all(d.xtr, b, sd), block_shuffle_all(d.xva, b, sd + 1),
                      block_shuffle_all(d.xte, b, sd + 2))
        runs = []
        for s in [0] + list(q1["extra_seeds"]):
            m, h = run_or_load(cfg, f"q2_block{b}_cnn_lr{cnn['lr']:g}_s{s}", build_cnn, db.xtr, db.ytr, db.xva,
                               db.yva, **tkw(cfg, q1["epochs"], cnn["lr"], s))
            lg = predict(m, db.xte)
            runs.append({"seed": s, "best_val_acc": h["best_val_acc"], "test": test_metrics(lg, db.yte)})
            if s == 0:
                conf_store[f"block{b}_cnn"] = lg.softmax(1).cpu().numpy()
        out["blocks"][str(b)] = {"runs": runs, "test_acc_mean": float(np.mean([r["test"]["acc"] for r in runs])),
                                 "test_acc_std": float(np.std([r["test"]["acc"] for r in runs]))}
    np.savez_compressed(rdir(cfg, "q2_test_probs.npz"), y=d.yte.cpu().numpy(), **conf_store)
    save_json(out, rdir(cfg, "q2.json"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    run(load_cfg(smoke=ap.parse_args().smoke))
