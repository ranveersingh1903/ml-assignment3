"""Turns results/*.json into every figure (figures/*.pdf), every table (report/generated/*.tex)
and a macro file of headline numbers (report/generated/numbers.tex). Nothing in the report
is typed by hand: re-running this script after an experiment updates the PDF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, CLASSES, load_cfg, load_json, rdir

# validated categorical order (blue, orange, aqua, yellow, magenta); text stays in ink colours
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.6,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 200})
COL_W, FULL_W = 3.3, 6.9   # CVPR column / text widths in inches

SHORT = {"airplane": "plane", "automobile": "car", "bird": "bird", "cat": "cat", "deer": "deer", "dog": "dog",
         "frog": "frog", "horse": "horse", "ship": "ship", "truck": "truck"}


def sn(c):
    return SHORT[CLASSES[c]]


MACROS: dict[str, str] = {}


def mac(name, value, fmt="{:.1f}"):
    MACROS[name] = fmt.format(value) if not isinstance(value, str) else value


def pct(x):
    return 100 * x


def tex_table(path, header, rows, caption, label, align=None, wide=False, note=None):
    align = align or "l" + "r" * (len(header) - 1)
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
             "\\begin{adjustbox}{max width=\\linewidth}", f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if note:
        lines.append(f"\\\\[2pt]\\parbox{{\\linewidth}}{{\\scriptsize {note}}}")
    lines += [f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\end{{{env}}}"]
    path.write_text("\n".join(lines) + "\n")


def ms(mean, std):
    return f"{pct(mean):.1f}$\\pm${pct(std):.1f}"


def fmt_params(n):
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


# ============================================================================ Q1
def q1(cfg, G, F):
    a = load_json(rdir(cfg, "q1a.json"))
    names = {"mlp": "MLP (1 hidden)", "mlp_deep": "Deep MLP", "patch_mlp": "Patch-MLP"}
    rows = []
    for k, m in a["models"].items():
        rows.append([names[k], fmt_params(m["params"]), f"{m['lr']:g}", f"{pct(m['train_acc_at_best_mean']):.1f}",
                     f"{pct(m['val_acc_mean']):.1f}", ms(m["test_acc_mean"], m["test_acc_std"])])
        mac(f"QoneTest{k.replace('_', '')}", pct(m["test_acc_mean"]))
        mac(f"QoneVal{k.replace('_', '')}", pct(m["val_acc_mean"]))
        mac(f"QoneTrain{k.replace('_', '')}", pct(m["train_acc_at_best_mean"]))
        mac(f"QoneParams{k.replace('_', '')}", fmt_params(m["params"]), None)
    for k, m in a["permuted"].items():
        rows.append([names[k] + " / permuted px", "same", "same", "--",
                     f"{pct(np.mean([r['best_val_acc'] for r in m['runs']])):.1f}", ms(m["test_acc_mean"], m["test_acc_std"])])
        mac(f"QonePermTest{k.replace('_', '')}", pct(m["test_acc_mean"]))
    tex_table(G / "q1a.tex", ["Model", "Params", "LR", "Train@best", "Val", "Test (mean$\\pm$sd)"], rows,
              "Q1(a) equal-budget comparison. All models matched to the same parameter budget ($\\pm$2\\%), "
              f"{cfg['q1']['epochs']} epochs, AdamW, cosine schedule, identical LR grid; LR chosen on val (seed 0), "
              f"then {1 + len(cfg['q1']['extra_seeds'])} seeds. Train@best = training accuracy at the val-selected epoch. "
              "Permuted rows use one fixed pixel permutation for train/val/test with the clean-selected LR.",
              "tab:q1a")
    spec_rows = [[names[k], f"\\texttt{{{m['spec']}}}", f"{m['params']:,}"] for k, m in a["models"].items()]
    tex_table(G / "q1a_spec.tex", ["Model", "Architecture", "\\#Params"], spec_rows,
              "Q1 model specifications (BatchNorm after every hidden Linear; LayerNorm inside the Patch-MLP token MLP).",
              "tab:q1spec", align="llr", wide=True)
    cap = a["capacity_sweep"]
    tex_table(G / "q1a_capacity.tex", ["Width", "Params", "Train@best", "Final train", "Val"],
              [[c["width"], fmt_params(c["params"]), f"{pct(c['train_acc_at_best']):.1f}",
                f"{pct(c['final_train_acc']):.1f}", f"{pct(c['best_val_acc']):.1f}"] for c in cap],
              "Q1(a) diagnostic: width sweep of the 1-hidden-layer MLP (parameters vary by design; validation only).",
              "tab:q1cap")
    mac("QoneCapValGain", pct(cap[-1]["best_val_acc"] - cap[0]["best_val_acc"]))
    mac("QoneCapTrainGain", pct(cap[-1]["final_train_acc"] - cap[0]["final_train_acc"]))

    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.2))
    for i, (k, m) in enumerate(a["models"].items()):
        h = load_json(rdir(cfg, "runs", f"{m['runs'][0]['run']}.json"))
        ax[0].plot(h["epoch"], pct(np.array(h["val_acc"])), color=C[i], label=f"{names[k]} val")
        ax[0].plot(h["epoch"], pct(np.array(h["train_acc"])), color=C[i], ls="--", lw=1.1, label=f"{names[k]} train")
    ax[0].set(xlabel="epoch", ylabel="accuracy (%)", title="(a) Equal-budget models: train (dashed) vs val (solid)")
    ax[0].legend(ncol=2, loc="lower right")
    w = [c["width"] for c in cap]
    ax[1].plot(w, [pct(c["final_train_acc"]) for c in cap], "o--", color=C[0], ms=4, label="final train")
    ax[1].plot(w, [pct(c["best_val_acc"]) for c in cap], "o-", color=C[0], ms=4, label="best val")
    ax[1].set(xscale="log", xlabel="hidden width (1 layer)", ylabel="accuracy (%)", title="(b) MLP capacity sweep")
    ax[1].set_xticks(w, [str(x) for x in w])
    ax[1].minorticks_off()
    ax[1].legend()
    fig.savefig(F / "q1a_curves.pdf")
    plt.close(fig)

    # --------------------------------------------------------------- Q1(b)
    b = load_json(rdir(cfg, "q1b.json"))
    rate = f"{cfg['q1']['memorisation']['noise_rate']:g}"
    h, h0 = b[rate], b["0"]
    ep = np.array(h["epoch"])
    val, nf, nt, cs = map(np.array, (h["val_acc"], h["noisy_fit"], h["noisy_true"], h["clean_subset_acc"]))
    pk = int(np.argmax(val))
    onset20 = int(ep[np.argmax(nf > 0.2)]) if (nf > 0.2).any() else None
    onset50 = int(ep[np.argmax(nf > 0.5)]) if (nf > 0.5).any() else None
    z = np.load(rdir(cfg, f"q1_mem_noise{rate}_fit.npz"))
    fit, noisy = z["fit"], z["noisy"]
    stays = np.flip(np.cumprod(np.flip(fit, 0), 0), 0).astype(bool)      # fitted from epoch e to the end
    learned_clean = stays[:, ~noisy].mean(1)
    learned_noisy = stays[:, noisy].mean(1)
    rows = [
        ["Peak validation", ep[pk], f"{pct(val[pk]):.1f}", f"{pct(cs[pk]):.1f}", f"{pct(nf[pk]):.1f}", f"{pct(nt[pk]):.1f}"],
    ]
    if onset50:
        i = onset50 - 1
        rows.append(["Noisy fit $>$50\\%", onset50, f"{pct(val[i]):.1f}", f"{pct(cs[i]):.1f}", f"{pct(nf[i]):.1f}", f"{pct(nt[i]):.1f}"])
    rows.append(["Final", ep[-1], f"{pct(val[-1]):.1f}", f"{pct(cs[-1]):.1f}", f"{pct(nf[-1]):.1f}", f"{pct(nt[-1]):.1f}"])
    tex_table(G / "q1b.tex", ["Point in training", "Epoch", "Val", "Clean-subset", "Noisy: fit noisy", "Noisy: true"], rows,
              f"Q1(b) memorisation: {fmt_params(h['params'])}-parameter MLP, {int(float(rate) * 100)}\\% of training labels replaced "
              "by a random wrong class, no weight decay, constant LR. \\emph{Noisy: fit noisy} = fraction of corrupted examples "
              "predicted as their (wrong) training label; \\emph{Noisy: true} = fraction predicted as their hidden true class. "
              f"Clean-label control (same model/protocol): peak val {pct(max(h0['val_acc'])):.1f}\\%, final {pct(h0['val_acc'][-1]):.1f}\\%. "
              f"Test accuracy of the val-selected checkpoint: {pct(h['test_acc_best_val_ckpt']):.1f}\\% (noisy) vs "
              f"{pct(h0['test_acc_best_val_ckpt']):.1f}\\% (clean).", "tab:q1b")
    mac("QoneBPeakEpoch", str(int(ep[pk])), None)
    mac("QoneBPeakVal", pct(val[pk]))
    mac("QoneBPeakNoisyFit", pct(nf[pk]))
    mac("QoneBPeakNoisyTrue", pct(nt[pk]))
    mac("QoneBFinalVal", pct(val[-1]))
    mac("QoneBFinalNoisyFit", pct(nf[-1]))
    mac("QoneBFinalNoisyTrue", pct(nt[-1]))
    mac("QoneBOnsetFifty", str(onset50) if onset50 else "--", None)
    mac("QoneBOnsetTwenty", str(onset20) if onset20 else "--", None)
    mac("QoneBParams", fmt_params(h["params"]), None)
    mac("QoneBTest", pct(h["test_acc_best_val_ckpt"]))
    mac("QoneBCleanCtrlFinal", pct(h0["val_acc"][-1]))
    mac("QoneBCleanCtrlPeak", pct(max(h0["val_acc"])))

    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.2))
    ax[0].plot(ep, pct(cs), color=C[0], label="train, clean subset")
    ax[0].plot(ep, pct(nf), color=C[1], label="train, noisy subset = noisy label (memorised)")
    ax[0].plot(ep, pct(nt), color=C[2], label="train, noisy subset = TRUE label")
    ax[0].plot(ep, pct(val), color=C[3], label="validation (clean)")
    ax[0].plot(h0["epoch"], pct(np.array(h0["val_acc"])), color=MUTED, ls=":", lw=1.2, label="validation, clean-label control")
    ax[0].axvline(ep[pk], color=MUTED, lw=0.8, ls="--")
    ax[0].set(xlabel="epoch", ylabel="accuracy / agreement (%)", title="(a) Learning vs memorisation across training")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, -0.25))
    ax[1].plot(ep, pct(learned_clean), color=C[0], label="clean examples")
    ax[1].plot(ep, pct(learned_noisy), color=C[1], label="noisy examples")
    ax[1].set(xlabel="epoch", ylabel="% fitted from here to the end", title="(b) When examples are permanently fitted")
    ax[1].legend()
    fig.savefig(F / "q1b_memorisation.pdf")
    plt.close(fig)


# ============================================================================ Q2
def q2(cfg, G, F):
    r = load_json(rdir(cfg, "q2.json"))
    m, c = r["models"]["mlp"], r["models"]["cnn"]
    rows = [["MLP (reused Q1)", fmt_params(m["params"]), f"{m['macs'] / 1e6:.1f}M",
             ms(m["test_acc_mean"], m["test_acc_std"]), ms(r["perm"]["mlp"]["test_acc_mean"], r["perm"]["mlp"]["test_acc_std"])]
            + ["--"] * len(r["blocks"]),
            ["CNN", fmt_params(c["params"]), f"{c['macs'] / 1e6:.0f}M", ms(c["test_acc_mean"], c["test_acc_std"]),
             ms(r["perm"]["cnn"]["test_acc_mean"], r["perm"]["cnn"]["test_acc_std"])]
            + [ms(v["test_acc_mean"], v["test_acc_std"]) for v in r["blocks"].values()]]
    hdr = ["Model", "Params", "MACs", "Clean", "Pixel perm."] + [f"Block {b}px" for b in r["blocks"]]
    tex_table(G / "q2.tex", hdr, rows,
              "Q2 test accuracy (\\%, mean$\\pm$sd over seeds) for the clean, spatially disturbed (fixed pixel permutation) "
              "and third (per-image block shuffle) conditions. Each condition is retrained with the transformation applied "
              f"identically to train/val/test. Specs: MLP \\texttt{{{m['spec']}}}; CNN \\texttt{{{c['spec']}}}.",
              "tab:q2", wide=True)
    mac("QtwoMLPclean", pct(m["test_acc_mean"]))
    mac("QtwoCNNclean", pct(c["test_acc_mean"]))
    mac("QtwoMLPperm", pct(r["perm"]["mlp"]["test_acc_mean"]))
    mac("QtwoCNNperm", pct(r["perm"]["cnn"]["test_acc_mean"]))
    mac("QtwoCNNparams", fmt_params(c["params"]), None)
    mac("QtwoMLPparams", fmt_params(m["params"]), None)
    mac("QtwoMACratio", c["macs"] / m["macs"], "{:.0f}")
    for b, v in r["blocks"].items():
        mac(f"QtwoBlock{['', 'A', 'B', 'C', 'D'][list(r['blocks']).index(b) + 1]}", pct(v["test_acc_mean"]))

    # confidence / error table (seed 0 runs + seed means)
    conds = [("Clean", c["runs"]), ("Pixel perm.", r["perm"]["cnn"]["runs"])] + \
            [(f"Block {b}px", v["runs"]) for b, v in r["blocks"].items()]
    rows = []
    for name, runs in conds:
        t = [x["test"] for x in runs]
        rows.append([name, f"{pct(np.mean([x['acc'] for x in t])):.1f}", f"{np.mean([x['nll'] for x in t]):.2f}",
                     f"{pct(np.mean([x['conf_correct'] for x in t])):.1f}", f"{pct(np.mean([x['conf_wrong'] for x in t])):.1f}",
                     f"{pct(np.mean([x['ece'] for x in t])):.1f}"])
    tex_table(G / "q2_conf.tex", ["CNN condition", "Acc", "NLL", "Conf$|$correct", "Conf$|$wrong", "ECE"], rows,
              "Q2(c) CNN test errors and confidence (means over seeds); each condition trained and tested with the same "
              "transformation. Conf$|$correct / Conf$|$wrong = mean max-softmax on correct / wrong test predictions.", "tab:q2conf")
    # per-class figure
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.3))
    x = np.arange(10)
    series = [("clean", c["runs"]), ("block " + list(r["blocks"])[1] + "px" if len(r["blocks"]) > 1 else "block",
                                     list(r["blocks"].values())[min(1, len(r["blocks"]) - 1)]["runs"]),
              ("block " + list(r["blocks"])[-1] + "px", list(r["blocks"].values())[-1]["runs"]),
              ("pixel perm.", r["perm"]["cnn"]["runs"])]
    for i, (lab, runs) in enumerate(series):
        pc = np.mean([x_["test"]["per_class_acc"] for x_ in runs], 0)
        ax[0].plot(x, pct(pc), "o-", color=C[i], ms=4, label=lab)
    ax[0].set_xticks(x, CLASSES, rotation=45, ha="right")
    ax[0].set(ylabel="test accuracy (%)", title="(a) CNN per-class accuracy by condition")
    ax[0].legend(ncol=4, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    z = np.load(rdir(cfg, "q2_test_probs.npz"))
    y = z["y"]
    keys = [k for k in ["clean_cnn"] + [f"block{b}_cnn" for b in r["blocks"]] + ["perm_cnn"] if k in z]
    bins = np.linspace(0.1, 1, 10)
    for i, k in enumerate(keys[:5]):
        p = z[k]
        conf, pred = p.max(1), p.argmax(1)
        accs = [((pred == y)[(conf > lo) & (conf <= hi)]).mean() if ((conf > lo) & (conf <= hi)).any() else np.nan
                for lo, hi in zip(bins[:-1], bins[1:])]
        ax[1].plot((bins[:-1] + bins[1:]) / 2, accs, "o-", ms=3.5, color=C[i], label=k.replace("_cnn", ""))
    ax[1].plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    ax[1].set(xlabel="confidence", ylabel="accuracy", title="(b) Reliability (seed 0)", xlim=(0.1, 1), ylim=(0, 1))
    ax[1].legend()
    fig.savefig(F / "q2_analysis.pdf")
    plt.close(fig)


# ============================================================================ Q3
def q3(cfg, G, F):
    r = load_json(rdir(cfg, "q3.json"))
    sel = r["selected_bottleneck"]
    rows = [[f"d={b}" + (" (selected)" if int(b) == sel else ""), f"{v['params']:,}", f"{v['best_val_mse']:.5f}",
             f"{v['val_psnr']:.2f}", v["best_epoch"]] for b, v in r["aes"].items()]
    tex_table(G / "q3a.tex", ["Bottleneck", "Params", "Val MSE", "Val PSNR", "Best ep."], rows,
              f"Q3(a) autoencoders differing only in the Linear bottleneck. Selected by lowest val MSE; its test MSE is "
              f"{r['selected_test_mse']:.5f} (PSNR {r['selected_test_psnr']:.2f} dB).", "tab:q3a")
    mac("QthreeSel", str(sel), None)
    mac("QthreeTestMSE", r["selected_test_mse"], "{:.5f}")
    mac("QthreeTestPSNR", r["selected_test_psnr"], "{:.2f}")
    cl = r["classification"]
    names = {"clean": "Clean (reference)", "noisy": "Noisy input", "ae(noisy)": f"Clean AE (d={sel}) on noisy",
             "dae(noisy)": f"DAE (d={sel}) on noisy", "ae(clean)": "Clean AE on clean", "dae(clean)": "DAE on clean"}
    rows = [[names[k], f"{pct(v['acc']):.1f}", f"{pct(v['mean_conf']):.1f}", f"{pct(v['ece']):.1f}",
             f"{v['psnr_to_clean']:.2f}" if v["mse_to_clean"] > 0 else "$\\infty$"] for k, v in cl.items()]
    tex_table(G / "q3b.tex", ["Input to fixed CNN", "Acc", "Mean conf", "ECE", "PSNR vs clean"], rows,
              f"Q3(b) test classification with the fixed Q2(a) CNN (seed 0). Corruption: Gaussian noise $\\sigma={r['dae']['sigma']}$. "
              f"Recovered fraction of the clean--noisy gap by the DAE: {pct(r['recovered_fraction']):.1f}\\%.", "tab:q3b")
    for k, short in (("clean", "Clean"), ("noisy", "Noisy"), ("ae(noisy)", "AEnoisy"), ("dae(noisy)", "DAEnoisy"),
                     ("ae(clean)", "AEclean"), ("dae(clean)", "DAEclean")):
        mac(f"Qthree{short}Acc", pct(cl[k]["acc"]))
        mac(f"Qthree{short}PSNR", cl[k]["psnr_to_clean"], "{:.2f}")
    mac("QthreeRecovered", pct(r["recovered_fraction"]))
    rows = [[k, v["dim"], f"{v['test_mse_clean']:.5f}", f"{pct(v['linear']['val_acc']):.1f}",
             f"{pct(v['linear']['test_acc']):.1f}", f"{pct(v['knn_test_acc']):.1f}"] for k, v in r["probe"].items()]
    tex_table(G / "q3c.tex", ["Frozen encoder", "Dim", "Test MSE", "Probe val", "Probe test", f"{cfg['q3']['knn_k']}-NN test"], rows,
              "Q3(c) class information in frozen bottleneck features (linear probe = L2 softmax regression, strength chosen on "
              "val) vs clean-image test reconstruction error.", "tab:q3c")
    for i, (k, v) in enumerate(r["probe"].items()):   # QthreeProbeA.. in table order
        mac(f"QthreeProbe{'ABCDEFG'[i]}", pct(v["linear"]["test_acc"]))
        mac(f"QthreeProbeName{'ABCDEFG'[i]}", k, None)
    # figure: examples + probe-vs-mse
    z = np.load(rdir(cfg, "q3_examples.npz"))
    keys = ["clean", "noisy", "ae_noisy", "dae_noisy"]
    labels = ["clean", "noisy", "clean AE(noisy)", "DAE(noisy)"]
    n = 8
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.0), gridspec_kw={"width_ratios": [1.6, 1]})
    grid = np.concatenate([np.concatenate([z[k][i].transpose(1, 2, 0) for i in range(n)], 1) for k in keys], 0)
    ax[0].imshow(np.clip(grid, 0, 1))
    ax[0].set_yticks([16 + 32 * i for i in range(4)], labels)
    ax[0].set_xticks([])
    ax[0].grid(False)
    ax[0].set_title("(a) Test examples")
    pts = [(k, v["test_mse_clean"], pct(v["linear"]["test_acc"])) for k, v in r["probe"].items()]
    for i, (k, x_, y_) in enumerate(pts):
        ax[1].scatter(x_, y_, color=C[i % 5], s=28, zorder=3, edgecolor="white", linewidth=1.5)
        ax[1].annotate(k, (x_, y_), textcoords="offset points", xytext=(4, 3), fontsize=6, color=INK)
    ax[1].set(xlabel="test reconstruction MSE", ylabel="linear probe test acc (%)",
              title="(b) Reconstruction vs class information")
    fig.savefig(F / "q3.pdf")
    plt.close(fig)


# ============================================================================ Q4
def q4(cfg, G, F, d):
    r = load_json(rdir(cfg, "q4.json"))
    sw = r["sweep"]
    rows = [[s["kernel"], f"{s['C']:g}", f"{pct(s['train_acc']):.1f}", f"{pct(s['val_acc']):.1f}",
             f"{pct(s['sv_frac']):.1f}", f"{s['fit_time_s']:.0f}", "" if s["converged"] else "$\\dagger$"] for s in sw]
    tex_table(G / "q4a.tex", ["Kernel", "C", "Train", "Val", "SV \\%", "Fit s", ""], rows,
              f"Q4(a) full C$\\times$kernel sweep on the fixed representation \\emph{{{r['representation']}}} "
              f"({r['dim']}-d, standardised), fixed stratified subset of {r['train_subset']} training images, full validation set. "
              f"Selected: {r['selected']['kernel']}, C={r['selected']['C']:g} (val {pct(r['selected']['val_acc']):.1f}\\%); "
              f"test accuracy {pct(r['test_acc']):.1f}\\%. $\\dagger$ = hit the iteration cap.", "tab:q4a")
    mac("QfourRep", r["representation"], None)
    mac("QfourTest", pct(r["test_acc"]))
    mac("QfourSelKernel", r["selected"]["kernel"], None)
    mac("QfourSelC", f"{r['selected']['C']:g}", None)
    mac("QfourSubset", f"{r['train_subset']:,}", None)
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.1))
    for i, k in enumerate(cfg["q4"]["kernels"]):
        s = [x for x in sw if x["kernel"] == k]
        Cs = [x["C"] for x in s]
        ax[0].plot(Cs, [pct(x["val_acc"]) for x in s], "o-", color=C[i], ms=4, label=f"{k} val")
        ax[0].plot(Cs, [pct(x["train_acc"]) for x in s], "o--", color=C[i], ms=3, lw=1, label=f"{k} train")
        ax[1].plot(Cs, [pct(x["sv_frac"]) for x in s], "o-", color=C[i], ms=4, label=k)
    ax[0].set(xscale="log", xlabel="C", ylabel="accuracy (%)", title="(a) Train (dashed) and val (solid) vs C")
    ax[0].legend(ncol=2, fontsize=6)
    ax[1].set(xscale="log", xlabel="C", ylabel="support vectors (% of train)", title="(b) Support-vector fraction vs C")
    ax[1].legend()
    fig.savefig(F / "q4a.pdf")
    plt.close(fig)

    # pairs
    rows = []
    for p in r["pairs"]:
        rows.append([f"{p['classes'][0]}/{p['classes'][1]}", p["test_confusions"], f"{pct(p['pair_test_err_rate']):.1f}",
                     f"{pct(p['sv_frac_of_pair_train']):.0f}", f"{pct(p['frac_sv_at_bound']):.0f}",
                     f"{pct(p['frac_pair_errors_inside_margin']):.0f}", f"{p['mean_abs_f_errors']:.2f}/{p['mean_abs_f_correct']:.2f}",
                     f"{p['fisher_ratio']:.3f}", f"{pct(p['knn10_pair_purity']):.0f}"])
    tex_table(G / "q4b.tex", ["Pair", "Conf.", "Err\\%", "SV\\%", "@C\\%", "Err in margin\\%", "$|f|$ err/corr", "Fisher", "10NN pur."],
              rows, "Q4(b) the most-confused test pairs for the selected SVM. SV\\% = share of the pair's training points "
              "that are support vectors of the pair's own OvO classifier; @C = share of those at the box bound "
              "(margin violators); Err in margin = share of pair errors with $|f_{ij}|<1$; Fisher = between-mean distance$^2$ "
              "over summed within-class variance; 10NN pur. = same-class share of 10 cosine nearest neighbours.",
              "tab:q4b", wide=True)
    # image grid of SVs and near-boundary test points
    xtr, xte = d.xtr.cpu().numpy(), d.xte.cpu().numpy()
    npairs = len(r["pairs"])
    nshow = 6
    fig, ax = plt.subplots(npairs * 2, nshow, figsize=(FULL_W * 0.8, 1.3 * npairs * 2))
    for pi, p in enumerate(r["pairs"]):
        svs = sorted(p["support_vectors"], key=lambda s: -abs(s["alpha"]))
        sv_pick = [s for s in svs if s["label"] == p["ids"][0]][:nshow // 2] + [s for s in svs if s["label"] == p["ids"][1]][:nshow // 2]
        for j, s in enumerate(sv_pick):
            a = ax[2 * pi, j]
            a.imshow(xtr[s["train_idx"]].transpose(1, 2, 0))
            a.set_title(f"SV {sn(s['label'])}\nf={s['f']:+.2f}", fontsize=5.5, color=INK, pad=2)
        for j, t in enumerate(p["near_boundary_test"][:nshow]):
            a = ax[2 * pi + 1, j]
            a.imshow(xte[t["test_idx"]].transpose(1, 2, 0))
            wrong = "x" if t["pred"] != t["label"] else ""
            a.set_title(f"{sn(t['label'])}->{sn(t['pred'])}{' (err)' if wrong else ''}\nf={t['f']:+.3f}", fontsize=5.5, color=INK, pad=2)
        ax[2 * pi, 0].set_ylabel(f"{sn(p['ids'][0])}/{sn(p['ids'][1])}\ntrain SVs", fontsize=6)
        ax[2 * pi + 1, 0].set_ylabel("nearest test", fontsize=6)
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    fig.suptitle("$f_{ij}>0$ votes for the first class of the pair; \"a->b\" = true->predicted (10-class prediction)", fontsize=6.5, color=MUTED)
    fig.savefig(F / "q4b_pairs.pdf")
    plt.close(fig)

    # clustering
    cs = r["cluster_selection"]
    rows = [[s["method"], s["k"], f"{s['val_silhouette']:.3f}", f"{s.get('val_bic', float('nan')):.3g}" if "val_bic" in s else "--",
             f"{s['val_nmi_reference_only']:.3f}"] for s in cs]
    tex_table(G / "q4c_sel.tex", ["Method", "k", "Val silhouette", "Val BIC", "Val NMI (not used)"], rows,
              "Q4(c) clustering model selection on validation representations (fit on training representations). "
              "The rule fixed in advance is maximum validation silhouette (label-free); NMI is shown only for reference.",
              "tab:q4csel")
    ct = r["cluster_test"]
    mac("QfourARI", ct["ARI"], "{:.3f}")
    mac("QfourNMI", ct["NMI"], "{:.3f}")
    mac("QfourClu", f"{r['cluster_selected']['method']}, k={r['cluster_selected']['k']}", None)
    mac("QfourMapAcc", pct(ct["majority_map_acc"]))
    mac("QfourPurity", pct(ct["purity"]))
    mac("QfourSpearman", r["spearman_svm_vs_cluster_pairs"], "{:.2f}")
    md = r["max_disagreement_pair"]
    mac("QfourDisPair", f"{md['pair'][0]}/{md['pair'][1]}", None)
    mac("QfourDisSVM", pct(md["svm_confusion_rate"]))
    mac("QfourDisCo", pct(md["cluster_coassignment"]))
    cont = np.array(ct["contingency"])
    fig, ax = plt.subplots(figsize=(COL_W, 0.25 * cont.shape[0] + 0.9))
    norm = cont / cont.sum(0, keepdims=True)
    im = ax.imshow(norm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f7f9fd", "#86b6ef", "#2a78d6", "#0d366b"]),
                   aspect="auto", vmin=0, vmax=1)
    for i in range(cont.shape[0]):
        for j in range(10):
            ax.text(j, i, cont[i, j], ha="center", va="center", fontsize=5, color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(cont.shape[0]), [f"c{i}" for i in range(cont.shape[0])])
    ax.grid(False)
    ax.set_title(f"Test cluster x class counts (shade = p(cluster|class)); ARI {ct['ARI']:.3f}, NMI {ct['NMI']:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.savefig(F / "q4c_contingency.pdf")
    plt.close(fig)
    pc = r["per_class"]
    tex_table(G / "q4c_perclass.tex", ["Class", "SVM recall", "Max p(cluster$|$class)", "Dominant cluster"],
              [[x["class"], f"{pct(x['svm_recall']):.1f}", f"{pct(x['cluster_concentration']):.1f}", f"c{x['dominant_cluster']}"] for x in pc],
              f"Q4(c) per-class comparison on the test set. Pair-level Spearman correlation between SVM confusion rate and "
              f"cluster co-assignment over all 45 pairs: {r['spearman_svm_vs_cluster_pairs']:.2f}. Largest disagreement: "
              f"{md['pair'][0]}/{md['pair'][1]} (SVM confusion {pct(md['svm_confusion_rate']):.1f}\\% vs co-assignment "
              f"{pct(md['cluster_coassignment']):.1f}\\%).", "tab:q4cpc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(smoke=a.smoke)
    G = ROOT / "report" / ("generated_smoke" if a.smoke else "generated")
    F = ROOT / cfg["figures_dir"]
    G.mkdir(parents=True, exist_ok=True)
    F.mkdir(parents=True, exist_ok=True)
    from common import get_data
    d = get_data(cfg)
    for fn, args in ((q1, ()), (q2, ()), (q3, ()), (q4, (d,))):
        try:
            fn(cfg, G, F, *args)
            print(f"[assets] {fn.__name__} done")
        except FileNotFoundError as e:
            print(f"[assets] skipped {fn.__name__}: missing {e.filename}")
    mac("SplitSeed", str(cfg["seed"]), None)
    mac("RollNumber", cfg["roll_number"], None)
    mac("StudentName", cfg["name"], None)
    mac("RepoURL", "\\url{" + cfg["repo_url"] + "}", None)
    import subprocess
    try:
        h = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        h = ""
    mac("CommitHash", h or "UNCOMMITTED", None)
    from conclusions import build
    with open(G / "conclusions.tex", "w") as f:
        for k, v in build(cfg).items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(G / "numbers.tex", "w") as f:
        for k, v in sorted(MACROS.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"[assets] wrote {len(MACROS)} macros to {G / 'numbers.tex'}")


if __name__ == "__main__":
    main()
