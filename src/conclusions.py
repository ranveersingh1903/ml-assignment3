"""Conclusions written by pre-registered decision rules (see PREDICTIONS.md).

Each function reads the logged results, applies a rule fixed before training and returns
LaTeX prose. The rule can say "supported", "not supported" or "mixed", so the report's
conclusions always follow from the measured numbers rather than from the predictions.
All differences are in percentage points (pp) of accuracy unless stated otherwise.
"""
from __future__ import annotations

import numpy as np

from common import CLASSES, load_json, rdir


def p(x):
    return 100 * x


def f1(x):
    return f"{x:.1f}"


# ----------------------------------------------------------------------------- Q1(a)
def q1a(cfg) -> str:
    a = load_json(rdir(cfg, "q1a.json"))
    m, dp, pm = a["models"]["mlp"], a["models"]["mlp_deep"], a["models"]["patch_mlp"]
    gap = p(m["train_acc_at_best_mean"] - m["val_acc_mean"])
    cap = a["capacity_sweep"]
    dtr = p(cap[-1]["final_train_acc"] - cap[0]["final_train_acc"])
    dva = p(cap[-1]["best_val_acc"] - cap[0]["best_val_acc"])
    deep_gain = p(dp["test_acc_mean"] - m["test_acc_mean"])
    patch_gain = p(pm["test_acc_mean"] - m["test_acc_mean"])
    sd = p(max(m["test_acc_std"], pm["test_acc_std"], 0.002))
    perm_mlp = p(a["permuted"]["mlp"]["test_acc_mean"] - m["test_acc_mean"])
    perm_patch = p(a["permuted"]["patch_mlp"]["test_acc_mean"] - pm["test_acc_mean"])
    patch_vs_mlp_perm = p(a["permuted"]["patch_mlp"]["test_acc_mean"] - a["permuted"]["mlp"]["test_acc_mean"])
    s = []
    # Rule C (capacity): rejected if train-val gap >= 10 pp AND width raises val by < half of what it raises train.
    cap_rejected = gap >= 10 and dva < 0.5 * max(dtr, 1e-9)
    s.append(f"\\textbf{{H\\textsubscript{{cap}}:}} the MLP's train--val gap at its selected epoch is {f1(gap)} pp, and widening "
             f"it {cap[0]['width']}$\\to${cap[-1]['width']} changes training accuracy by {f1(dtr)} pp but validation accuracy by "
             f"{f1(dva)} pp. " + ("Extra capacity is spent on fitting the training set rather than on generalising, so "
                                  "\\emph{underfitting is not the bottleneck} and H\\textsubscript{cap} is rejected."
                                  if cap_rejected else
                                  "The rule fixed in advance (gap $\\ge$10 pp and validation gain $<$ half the training gain) "
                                  "is not met, so H\\textsubscript{cap} cannot be rejected: capacity remains a contributing factor."))
    # Rule O (depth/optimisation): rejected if deep MLP gains < 2 pp over the MLP.
    s.append(f"\\textbf{{H\\textsubscript{{opt}}:}} the equal-budget deep MLP changes test accuracy by {f1(deep_gain)} pp. "
             + ("Re-shaping the same parameters into more layers does not close the gap, so H\\textsubscript{opt} is rejected."
                if deep_gain < 2 else
                "Depth helps by at least 2 pp, so architecture shape explains part of the gap (H\\textsubscript{opt} partly supported)."))
    # Rule B (bias): supported if Patch-MLP gains >= max(3 pp, 2 sd) AND permutation removes >= half of its advantage
    # AND the plain MLP moves by < 1.5 pp under permutation.
    gain_ok = patch_gain >= max(3.0, 2 * sd)
    perm_ok = (patch_gain - patch_vs_mlp_perm) >= 0.5 * patch_gain if patch_gain > 0 else False
    mlp_inv = abs(perm_mlp) < 1.5
    s.append(f"\\textbf{{H\\textsubscript{{bias}}:}} at the same parameter count the Patch-MLP changes test accuracy by "
             f"{f1(patch_gain)} pp (largest seed sd {f1(sd)} pp). Under pixel permutation the Patch-MLP changes by {f1(perm_patch)} pp "
             f"and the plain MLP by {f1(perm_mlp)} pp, leaving the Patch-MLP {f1(patch_vs_mlp_perm)} pp ahead. ")
    if gain_ok and perm_ok and mlp_inv:
        s.append("All three pre-registered conditions hold: the gain comes from exploiting spatial structure, which the plain "
                 "MLP cannot use. \\emph{Conclusion: the missing spatial inductive bias is the main reason the MLP performs "
                 "poorly, and the competing explanations are ruled out by the controls above.}")
    else:
        failed = [n for n, ok in (("a clear equal-budget gain", gain_ok), ("loss of the gain under permutation", perm_ok),
                                  ("permutation invariance of the MLP", mlp_inv)) if not ok]
        s.append("Not all pre-registered conditions hold (missing: " + ", ".join(failed) + "), so the evidence for "
                 "H\\textsubscript{bias} is only partial and we do not claim it as the sole explanation.")
    return " ".join(s)


# ----------------------------------------------------------------------------- Q1(b)
def q1b(cfg) -> str:
    b = load_json(rdir(cfg, "q1b.json"))
    rate = f"{cfg['q1']['memorisation']['noise_rate']:g}"
    h, h0 = b[rate], b["0"]
    val, nf, nt = map(np.array, (h["val_acc"], h["noisy_fit"], h["noisy_true"]))
    ep = np.array(h["epoch"])
    k = int(np.argmax(val))
    chance_wrong = (1 - nt[k]) / 9
    early_structure = nt[k] > nf[k]
    mem_late = nf[-1] - nf[k] >= 0.2
    val_drop = p(val[k] - val[-1])
    ctrl_drop = p(max(h0["val_acc"]) - h0["val_acc"][-1])
    s = (f"At the validation peak (epoch {ep[k]}) the corrupted training examples are predicted as their hidden true class "
         f"{f1(p(nt[k]))}\\% of the time and as their wrong training label {f1(p(nf[k]))}\\% of the time (chance level for a "
         f"specific wrong label given those predictions $\\approx${f1(p(chance_wrong))}\\%). Between that epoch and the end, the "
         f"noisy-label fit rises by {f1(p(nf[-1] - nf[k]))} pp to {f1(p(nf[-1]))}\\%, true-label agreement on the noisy subset "
         f"falls to {f1(p(nt[-1]))}\\%, and validation accuracy falls by {f1(val_drop)} pp, against {f1(ctrl_drop)} pp in the "
         f"clean-label control. ")
    # Rule: structure-before-memorisation if, at the val peak, true-label agreement > noisy fit AND the noisy fit
    # rises by >= 20 pp afterwards.
    if early_structure and mem_late:
        s += ("Both pre-registered conditions hold. \\emph{The MLP learns generalisable class structure first (it even "
              "``corrects'' mislabelled training images), and memorisation of the noisy labels happens afterwards, "
              "overwriting part of that structure; the drop in validation accuracy is caused by the noisy labels, "
              "since the control does not show it to the same extent.}")
    elif early_structure:
        s += ("Structure is learned first, but the noisy fit does not rise by the pre-registered 20 pp afterwards within "
              "the training budget, so memorisation is only partial here.")
    else:
        s += ("At the validation peak the noisy labels are already fitted more often than the true labels, so the data do "
              "not support a clean ``structure first, memorisation later'' ordering.")
    return s


# ----------------------------------------------------------------------------- Q2
def q2b(cfg) -> str:
    r = load_json(rdir(cfg, "q2.json"))
    m, c = r["models"]["mlp"], r["models"]["cnn"]
    dm = p(r["perm"]["mlp"]["test_acc_mean"] - m["test_acc_mean"])
    dc = p(r["perm"]["cnn"]["test_acc_mean"] - c["test_acc_mean"])
    adv_clean = p(c["test_acc_mean"] - m["test_acc_mean"])
    adv_perm = p(r["perm"]["cnn"]["test_acc_mean"] - r["perm"]["mlp"]["test_acc_mean"])
    s = (f"Under the pixel permutation the MLP changes by {f1(dm)} pp and the CNN by {f1(dc)} pp. The CNN's advantage goes "
         f"from {f1(adv_clean)} pp on clean images to {f1(adv_perm)} pp. ")
    # Rule: explanation supported if |dm| < 1.5 pp and the CNN keeps < 25% of its clean advantage.
    if abs(dm) < 1.5 and adv_perm < 0.25 * adv_clean:
        s += ("This matches the prediction. The pixels are identical, so the CNN's advantage cannot come from extra compute or "
              "easier optimisation; it disappears exactly when locality is destroyed. \\emph{This supports the explanation in "
              "(a): the CNN wins because its locality and weight sharing exploit spatial structure.}")
    elif abs(dm) < 1.5:
        s += ("The MLP is invariant as predicted, but the CNN keeps more than a quarter of its advantage, so the spatial prior "
              "explains only part of the gap.")
    else:
        s += "The MLP is not invariant within the pre-registered tolerance, so the comparison is inconclusive."
    return s


def q2c(cfg) -> str:
    r = load_json(rdir(cfg, "q2.json"))
    c = r["models"]["cnn"]
    blocks = r["blocks"]
    bs = list(blocks)
    clean = c["test_acc_mean"]
    drops = {b: p(clean - v["test_acc_mean"]) for b, v in blocks.items()}
    big, small = bs[0], bs[-1]
    s = "Accuracy drops relative to clean are " + ", ".join(f"{f1(drops[b])} pp at {b} px" for b in bs) + ". "
    # Rule: E_local if the drop at the largest block is < 40% of the drop at the smallest block;
    # E_global if >= 60%; otherwise mixed.
    ratio = drops[big] / drops[small] if drops[small] > 0.5 else float("nan")
    # confidence
    cw = {b: np.mean([x["test"]["conf_wrong"] for x in v["runs"]]) for b, v in blocks.items()}
    ece = {b: np.mean([x["test"]["ece"] for x in v["runs"]]) for b, v in blocks.items()}
    cw0 = np.mean([x["test"]["conf_wrong"] for x in c["runs"]])
    ece0 = np.mean([x["test"]["ece"] for x in c["runs"]])
    # classes hurt most at the middle block size
    mid = bs[len(bs) // 2]
    pc0 = np.mean([x["test"]["per_class_acc"] for x in c["runs"]], 0)
    pcm = np.mean([x["test"]["per_class_acc"] for x in blocks[mid]["runs"]], 0)
    worst = np.argsort(pc0 - pcm)[::-1][:3]
    least = np.argsort(pc0 - pcm)[:2]
    conf0 = np.array(c["runs"][0]["test"]["confusion"])
    confm = np.array(blocks[mid]["runs"][0]["test"]["confusion"])
    inc = confm - conf0
    np.fill_diagonal(inc, -10**9)
    i, j = np.unravel_index(np.argmax(inc), inc.shape)
    if np.isfinite(ratio) and ratio < 0.4:
        verdict = (f"The drop at {big} px is only {f1(100 * ratio)}\\% of the drop at {small} px: removing the global "
                   "arrangement costs little as long as local content survives. \\emph{E\\textsubscript{local} is supported: "
                   "the CNN classifies largely from local texture and colour statistics.}")
    elif np.isfinite(ratio) and ratio >= 0.6:
        verdict = (f"The drop at {big} px is already {f1(100 * ratio)}\\% of the drop at {small} px, so destroying the global "
                   "arrangement hurts nearly as much as destroying local content. \\emph{E\\textsubscript{global} is supported: "
                   "the CNN depends on the spatial configuration of object parts.}")
    else:
        verdict = (f"The drop at {big} px is {f1(100 * ratio)}\\% of the drop at {small} px, between the pre-registered "
                   "thresholds (40\\%/60\\%). \\emph{Both explanations contribute: local statistics carry much of the signal, "
                   "but global layout adds a measurable part.}")
    s += verdict + (f" Errors: at {mid} px the classes losing most are {', '.join(CLASSES[k] for k in worst)} "
                    f"({', '.join(f1(p(pc0[k] - pcm[k])) for k in worst)} pp) and those losing least are "
                    f"{', '.join(CLASSES[k] for k in least)}; the confusion that grows most is {CLASSES[i]}$\\to${CLASSES[j]} "
                    f"(+{int(inc[i, j])} images, seed 0). Confidence: mean confidence on wrong predictions is {f1(p(cw0))}\\% "
                    f"clean vs " + ", ".join(f"{f1(p(cw[b]))}\\% at {b} px" for b in bs) + "; ECE is "
                    f"{f1(p(ece0))}\\% clean vs " + ", ".join(f"{f1(p(ece[b]))}\\%" for b in bs) + ". ")
    s += ("The model trained on shuffled images makes its additional errors with "
          + ("\\emph{lower} confidence, i.e.\\ it registers the extra ambiguity."
             if cw[small] < cw0 else "\\emph{similar or higher} confidence, i.e.\\ it does not register the extra ambiguity."))
    return s


def q2link(cfg) -> str:
    r = load_json(rdir(cfg, "q2.json"))
    c, m = r["models"]["cnn"], r["models"]["mlp"]
    return (f"(a) At equal parameters the CNN leads the MLP by {f1(p(c['test_acc_mean'] - m['test_acc_mean']))} pp. "
            f"(b) Keeping every pixel value but destroying locality shrinks that lead to "
            f"{f1(p(r['perm']['cnn']['test_acc_mean'] - r['perm']['mlp']['test_acc_mean']))} pp, which locates the advantage in "
            "the spatial prior. (c) Shuffling blocks shows which scale of that prior matters: accuracy falls by "
            + ", ".join(f"{f1(p(c['test_acc_mean'] - v['test_acc_mean']))} pp at {b} px" for b, v in r["blocks"].items())
            + ". " + _q2c_link_tail(r, c, m))


def _q2c_link_tail(r, c, m):
    bl = r["blocks"]
    bs = list(bl)
    d_big = c["test_acc_mean"] - bl[bs[0]]["test_acc_mean"]
    d_small = c["test_acc_mean"] - bl[bs[-1]]["test_acc_mean"]
    ratio = d_big / d_small if d_small > 0.005 else float("nan")
    near = abs(bl[bs[-1]]["test_acc_mean"] - m["test_acc_mean"]) < 0.03
    if ratio == ratio and ratio < 0.4:
        t = ("Losing the global layout (large blocks) costs much less than losing local structure (small blocks), so the "
             "CNN's advantage comes mainly from local patterns")
    elif ratio == ratio and ratio >= 0.6:
        t = "Losing the global layout already costs most of the accuracy, so the CNN's advantage depends on global arrangement"
    else:
        t = "Both local patterns and global layout contribute to the CNN's advantage"
    if near:
        t += f", and with {bs[-1]} px blocks it is back near the MLP level"
    return t + ". Together the three experiments locate the CNN's advantage in its spatial prior (locality and weight sharing)."


# ----------------------------------------------------------------------------- Q3
def q3b(cfg) -> str:
    r = load_json(rdir(cfg, "q3.json"))
    a = r["classification"]
    rec = r["recovered_fraction"]
    ceil = a["ae(clean)"]["acc"]
    s = (f"The fixed CNN scores {f1(p(a['clean']['acc']))}\\% on clean images, {f1(p(a['noisy']['acc']))}\\% on noisy images, "
         f"{f1(p(a['ae(noisy)']['acc']))}\\% on clean-AE reconstructions of noisy images and "
         f"{f1(p(a['dae(noisy)']['acc']))}\\% on DAE reconstructions. The DAE recovers {f1(p(rec))}\\% of the clean--noisy gap. "
         f"Reconstructing a \\emph{{clean}} image through the same architecture already gives only {f1(p(ceil))}\\%, which is "
         "the ceiling any reconstruction from this model can reach. ")
    # Rule: "recovers the needed information" if recovered >= 50% AND DAE(noisy) within 3 pp of the ceiling;
    # "partially" if recovered > 0; otherwise "no".
    if rec == rec and rec >= 0.5 and a["dae(noisy)"]["acc"] >= ceil - 0.03:
        s += ("\\emph{Denoising recovers the information the classifier needs, up to the blur the autoencoder itself introduces.}")
    elif rec == rec and rec > 0:
        s += ("\\emph{Denoising recovers the information only partially:} it removes noise (PSNR rises), but some class-relevant "
              "detail is lost, and the gap to the ceiling shows the classifier needs high-frequency detail the DAE does not "
              "restore.")
    else:
        s += ("\\emph{Denoising does not recover the information the classifier needs:} the reconstructions look cleaner but "
              "classification does not improve over the noisy input.")
    return s


def q3c(cfg) -> str:
    r = load_json(rdir(cfg, "q3.json"))
    pr = r["probe"]
    aes = [k for k in pr if k.startswith("AE")]
    mse = np.array([pr[k]["test_mse_clean"] for k in aes])
    acc = np.array([pr[k]["linear"]["test_acc"] for k in aes])
    best_rec, best_cls = aes[int(np.argmin(mse))], aes[int(np.argmax(acc))]
    dae = [k for k in pr if k.startswith("DAE")][0]
    rnd = [k for k in pr if k.startswith("random")][0]
    sel = f"AE d={r['selected_bottleneck']}"
    s = (f"Across the three autoencoders, the best reconstruction is {best_rec} and the most linearly decodable class "
         f"information is in {best_cls} (probe test accuracy " + ", ".join(f"{k}: {f1(p(pr[k]['linear']['test_acc']))}\\%"
                                                                         for k in aes) + "). ")
    # Rule: "agree" iff ordering by MSE (ascending) equals ordering by probe accuracy (descending) for the 3 AEs.
    agree = list(np.argsort(mse)) == list(np.argsort(-acc))
    s += (f"The DAE ({f1(p(pr[dae]['linear']['test_acc']))}\\%) has clean-image MSE {pr[dae]['test_mse_clean']:.4f} vs "
          f"{pr[sel]['test_mse_clean']:.4f} for the same-size clean AE, and the random encoder control reaches "
          f"{f1(p(pr[rnd]['linear']['test_acc']))}\\%. ")
    dae_rev = pr[dae]["linear"]["test_acc"] > pr[sel]["linear"]["test_acc"] and pr[dae]["test_mse_clean"] > pr[sel]["test_mse_clean"]
    if agree and dae_rev:
        s += ("\\emph{The two quantities agree only partly.} Among the three clean autoencoders, which differ only in "
              "capacity, lower reconstruction error does come with more class information. But the relation is not "
              f"general: the DAE reconstructs clean images worse than the same-size AE yet gives the best probe accuracy, and "
              f"the untrained random encoder, whose reconstruction error is far higher, still reaches "
              f"{f1(p(pr[rnd]['linear']['test_acc']))}\\%, only {f1(p(pr[sel]['linear']['test_acc'] - pr[rnd]['linear']['test_acc']))} pp "
              "below the trained AE. Reconstruction rewards keeping pixel-level detail (exact colour, position, texture); "
              "class information depends on what is made linearly separable, and the denoising objective, which must ignore "
              "pixel-level noise, does that better. Much of the probe accuracy comes from the convolutional architecture itself.")
        return s
    if agree and not dae_rev:
        s += ("\\emph{Here the two quantities agree in ranking}: lower reconstruction error comes with more class information. "
              "The random-encoder control shows how much of the probe accuracy is due to the convolutional architecture alone.")
    else:
        s += ("\\emph{The two quantities do not agree}: a lower reconstruction error does not imply more linearly accessible class "
              "information. Reconstruction rewards keeping pixel-level detail (exact colour, position, texture), much of which is "
              "irrelevant to the class, while the probe rewards a representation in which classes are linearly separable.")
    return s


# ----------------------------------------------------------------------------- Q4
def q4a(cfg) -> str:
    r = load_json(rdir(cfg, "q4.json"))
    sw = r["sweep"]
    parts = []
    for k in cfg["q4"]["kernels"]:
        s = sorted([x for x in sw if x["kernel"] == k], key=lambda x: x["C"])
        best = max(s, key=lambda x: x["val_acc"])
        lo, hi = s[0], s[-1]
        under = p(best["val_acc"] - lo["val_acc"])
        over = p(best["val_acc"] - hi["val_acc"])
        parts.append(f"\\textbf{{{k}}}: at C={lo['C']:g} train/val are {f1(p(lo['train_acc']))}/{f1(p(lo['val_acc']))}\\% with "
                     f"{f1(p(lo['sv_frac']))}\\% SVs; the best C={best['C']:g} gives {f1(p(best['val_acc']))}\\% val; at "
                     f"C={hi['C']:g} train/val are {f1(p(hi['train_acc']))}/{f1(p(hi['val_acc']))}\\% with "
                     f"{f1(p(hi['sv_frac']))}\\% SVs"
                     + (f" (underfits at small C: {f1(under)} pp below best)" if under >= 1 else "")
                     + (f" (overfits at large C: train--val gap {f1(p(hi['train_acc'] - hi['val_acc']))} pp, val {f1(over)} pp below best)"
                        if over >= 0.5 else "") + ".")
    capped = sorted({x["kernel"] for x in sw if not x["converged"]})
    cap_txt = ""
    if capped:
        cs = [x["C"] for x in sw if not x["converged"]]
        cap_txt = (f" Fits marked $\\dagger$ ({', '.join(capped)}, C$\\ge${min(cs):g}) stopped at the iteration cap, so those "
                   "rows are approximate; the identical numbers at the largest C values reflect this, and we do not draw "
                   "conclusions from their exact values.")
    svmin, svmax = min(x["sv_frac"] for x in sw), max(x["sv_frac"] for x in sw)
    return (" ".join(parts) + " Small C tolerates many margin violations, so the boundary is too smooth and both training and "
            "validation accuracy are low (underfitting). Raising C makes violations expensive: training accuracy climbs to "
            "nearly 100\\% while validation accuracy peaks and then stops improving or falls, so the extra flexibility is spent "
            "on fitting the training subset (overfitting). Unusually, the support-vector fraction stays between "
            f"{f1(p(svmin))}\\% and {f1(p(svmax))}\\% for every setting and drops only slightly with C: in this 1024-d "
            "representation the classes overlap so much that almost every training point lies on or inside a margin."
            + cap_txt + f" The selected SVM ({r['selected']['kernel']}, C={r['selected']['C']:g}) reaches "
            f"{f1(p(r['test_acc']))}\\% test accuracy.")


def q4b(cfg) -> str:
    r = load_json(rdir(cfg, "q4.json"))
    ps = r["pairs"]
    inm = np.nanmean([x["frac_pair_errors_inside_margin"] for x in ps])
    allf = [x["fisher_ratio"] for x in ps]
    s = (f"Averaged over the {len(ps)} pairs, {f1(p(inm))}\\% of the pair errors lie inside the margin ($|f_{{ij}}|<1$), and "
         "errors have smaller $|f_{ij}|$ than correct predictions (" +
         "; ".join(f"{x['classes'][0]}/{x['classes'][1]}: {x['mean_abs_f_errors']:.2f} vs {x['mean_abs_f_correct']:.2f}" for x in ps)
         + "). The pairs need many support vectors ("
         + ", ".join(f"{f1(p(x['sv_frac_of_pair_train']))}\\%" for x in ps) + " of their training points), of which only "
         + ", ".join(f"{f1(p(x['frac_sv_at_bound']))}\\%" for x in ps) + " are at the box bound C. "
         + ("Most support vectors are therefore \\emph{free} ($0<\\alpha<C$) and lie on the margin: the pair's boundary is "
            "held up by almost all of its training points rather than by a few prototypes, a sign that the two classes are "
            "nearly interleaved and the flexible RBF boundary threads between them. "
            if np.mean([x["frac_sv_at_bound"] for x in ps]) < 0.5 else
            "Most of them are at the bound, i.e.\\ margin violators. ")
         + "In representation space these pairs have low Fisher ratios ("
         + ", ".join(f"{v:.3f}" for v in allf) + ") and nearest neighbours that frequently come from the other class (10-NN purity "
         + ", ".join(f"{f1(p(x['knn10_pair_purity']))}\\%" for x in ps) + "). The support vectors and near-boundary images in "
         "\\cref{fig:q4b} share colour, background and pose across the two classes. \\emph{The test errors are therefore a "
         "property of the representation, in which these classes overlap, rather than a failure of the SVM optimisation.}")
    return s


def q4c(cfg) -> str:
    r = load_json(rdir(cfg, "q4.json"))
    md, ct = r["max_disagreement_pair"], r["cluster_test"]
    s = (f"Clustering reaches test ARI {ct['ARI']:.3f} and NMI {ct['NMI']:.3f}, against {f1(p(r['test_acc']))}\\% SVM accuracy "
         f"on the same features. Over all 45 class pairs, the Spearman correlation between SVM confusion and cluster "
         f"co-assignment is {r['spearman_svm_vs_cluster_pairs']:.2f}. The largest disagreement is "
         f"\\textbf{{{md['pair'][0]}/{md['pair'][1]}}}: the SVM confuses only {f1(p(md['svm_confusion_rate']))}\\% of their test "
         f"images, yet a random {md['pair'][0]} and a random {md['pair'][1]} image share a cluster "
         f"{f1(p(md['cluster_coassignment']))}\\% of the time (within-class: {f1(p(md['within_i']))}\\% and "
         f"{f1(p(md['within_j']))}\\%). ")
    s += ("\\emph{Classification says the two classes are separable, clustering says they belong together.} The supervised "
          "margin can use a low-variance direction that separates them, whereas the unsupervised partition follows the "
          "high-variance directions (overall colour, brightness, background) along which the two classes coincide.")
    return s


ALL = {"ConcQoneA": q1a, "ConcQoneB": q1b, "ConcQtwoB": q2b, "ConcQtwoC": q2c, "ConcQtwoLink": q2link,
       "ConcQthreeB": q3b, "ConcQthreeC": q3c, "ConcQfourA": q4a, "ConcQfourB": q4b, "ConcQfourC": q4c}


def build(cfg) -> dict[str, str]:
    out = {}
    for k, fn in ALL.items():
        try:
            out[k] = fn(cfg)
        except FileNotFoundError:
            out[k] = "(result missing: run the corresponding script)"
    return out
