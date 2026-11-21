"""Q4 (bonus) -- SVMs and clustering on ONE fixed representation.

Representation: chosen automatically, before any SVM is trained, as the Q3 encoder whose
frozen bottleneck features had the highest linear-probe VALIDATION accuracy (Q3c). It is
never changed afterwards.

(a) Fixed stratified training subset (cfg.q4.train_subset, drawn with the roll-number seed
    BEFORE model selection and saved to splits/). Full 5k validation set. Grid: kernels x C.
    Every setting records train acc, val acc, support-vector fraction, fit time, convergence.
    Selection on val accuracy only; the selected SVM is evaluated once on the test set.
(b) The n most-confused class pairs on the test set; for each, the pair's own one-vs-one
    decision function f_ij, its training support vectors (largest |alpha|) and the test
    images with the smallest |f_ij|; margin statistics of the errors; geometry of the pair in
    representation space (Fisher ratio, k-NN label purity).
(c) K-means vs diagonal GMM, k in cfg; chosen by mean silhouette on VALIDATION points
    (label-free) after fitting on training points. Final model fit on training
    representations, test points assigned without labels; ARI/NMI/contingency afterwards.
"""
from __future__ import annotations

import argparse
import time
import warnings
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import ROOT, CLASSES, load_cfg, get_data, save_json, load_json, rdir


def load_representation(cfg):
    q3 = load_json(rdir(cfg, "q3.json"))
    cands = {k: v["linear"]["val_acc"] for k, v in q3["probe"].items() if not k.startswith("random")}
    name = max(cands, key=cands.get) if cfg["q4"]["representation"] == "auto" else cfg["q4"]["representation"]
    z = np.load(rdir(cfg, "features", name.replace(" ", "_").replace("=", "") + ".npz"))
    return name, cands, z["tr"], z["va"], z["te"]


def get_subset(cfg, ytr):
    path = ROOT / ("splits/q4_subset_smoke.npz" if cfg["smoke"] else "splits/q4_subset.npz")
    if path.exists():
        return np.load(path)["idx"]
    idx, _ = train_test_split(np.arange(len(ytr)), train_size=cfg["q4"]["train_subset"], stratify=ytr,
                              random_state=cfg["seed"] % (2**32))
    idx = np.sort(idx)
    np.savez(path, idx=idx, seed=cfg["seed"])
    return idx


def fit_one(kernel, C, X, y, Xv, yv, cfg):
    q = cfg["q4"]
    clf = SVC(kernel=kernel, C=C, gamma="scale", degree=q["poly_degree"], coef0=1.0 if kernel == "poly" else 0.0,
              decision_function_shape="ovo", cache_size=2000, max_iter=q["max_iter"])
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X, y)
        converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    return {"kernel": kernel, "C": C, "train_acc": float((clf.predict(X) == y).mean()),
            "val_acc": float((clf.predict(Xv) == yv).mean()), "sv_frac": float(clf.n_support_.sum() / len(y)),
            "n_sv_per_class": clf.n_support_.tolist(), "fit_time_s": time.time() - t0, "converged": converged}


def ovo_pair_index(i, j, k=10):
    return [p for p in combinations(range(k), 2)].index((i, j))


def pair_dual(clf, i, j):
    """Support vectors and dual coefficients of the binary i-vs-j classifier inside sklearn's OvO SVC."""
    start = np.concatenate([[0], np.cumsum(clf.n_support_)])
    si, sj = np.arange(start[i], start[i + 1]), np.arange(start[j], start[j + 1])
    ai, aj = clf.dual_coef_[j - 1, si], clf.dual_coef_[i, sj]
    sv = np.concatenate([si, sj])
    a = np.concatenate([ai, aj])
    keep = np.abs(a) > 1e-12
    return clf.support_[sv[keep]], a[keep]


def run(cfg):
    d = get_data(cfg)
    q = cfg["q4"]
    ytr_all, yva, yte = d.ytr.cpu().numpy(), d.yva.cpu().numpy(), d.yte.cpu().numpy()
    rep, cands, ftr_all, fva, fte = load_representation(cfg)
    sub = get_subset(cfg, ytr_all)
    scaler = StandardScaler().fit(ftr_all[sub])
    X, y = scaler.transform(ftr_all[sub]), ytr_all[sub]
    Xv, Xt = scaler.transform(fva), scaler.transform(fte)
    out = {"representation": rep, "representation_candidates_val_probe": cands, "dim": int(X.shape[1]),
           "train_subset": int(len(sub)), "subset_class_counts": np.bincount(y, minlength=10).tolist()}

    # (a) ---------------------------------------------------------------------
    grid = [(k, C) for k in q["kernels"] for C in q["C_grid"]]
    sweep = Parallel(n_jobs=-1, verbose=5)(delayed(fit_one)(k, C, X, y, Xv, yva, cfg) for k, C in grid)
    out["sweep"] = sweep
    best = max(sweep, key=lambda r: (r["val_acc"], -r["sv_frac"]))
    out["selected"] = {"kernel": best["kernel"], "C": best["C"], "val_acc": best["val_acc"]}
    clf = SVC(kernel=best["kernel"], C=best["C"], gamma="scale", degree=q["poly_degree"],
              coef0=1.0 if best["kernel"] == "poly" else 0.0, decision_function_shape="ovo",
              cache_size=2000, max_iter=q["max_iter"]).fit(X, y)
    pred = clf.predict(Xt)
    out["test_acc"] = float((pred == yte).mean())
    cm = np.zeros((10, 10), int)
    np.add.at(cm, (yte, pred), 1)
    out["test_confusion"] = cm.tolist()
    out["test_per_class_recall"] = (cm.diagonal() / cm.sum(1)).tolist()

    # (b) ---------------------------------------------------------------------
    sym = cm + cm.T
    np.fill_diagonal(sym, 0)
    pairs = sorted([(int(sym[i, j]), i, j) for i in range(10) for j in range(i + 1, 10)], reverse=True)[:q["n_pairs"]]
    dec = clf.decision_function(Xt)
    out["pairs"] = []
    for n_conf, i, j in pairs:
        f = dec[:, ovo_pair_index(i, j)]
        # sklearn convention check: positive f_ij votes for class i
        m = np.isin(yte, [i, j])
        sign_ok = float(((f[m] > 0) == (yte[m] == i)).mean())
        sv_idx, alpha = pair_dual(clf, i, j)
        bounded = np.isclose(np.abs(alpha), best["C"], rtol=1e-3)
        order = np.argsort(-np.abs(alpha))
        sv_show = []
        for cls in (i, j):
            sel = [k for k in order if y[sv_idx[k]] == cls][:q["n_show"]]
            sv_show += [{"train_idx": int(sub[sv_idx[k]]), "label": cls, "alpha": float(alpha[k]),
                         "f": float(clf.decision_function(X[sv_idx[k]][None])[0, ovo_pair_index(i, j)])} for k in sel]
        test_ids = np.where(m)[0]
        near = test_ids[np.argsort(np.abs(f[test_ids]))[:2 * q["n_show"]]]
        err = m & np.isin(pred, [i, j]) & (pred != yte)
        # geometry in the (standardised) representation space
        A, B = X[y == i], X[y == j]
        mu_a, mu_b = A.mean(0), B.mean(0)
        fisher = float(np.sum((mu_a - mu_b) ** 2) / (A.var(0).sum() + B.var(0).sum()))
        P = np.concatenate([A, B])
        lab = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
        nrm = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        sims = nrm @ nrm.T
        np.fill_diagonal(sims, -np.inf)
        nn10 = np.argsort(-sims, axis=1)[:, :10]
        purity = float((lab[nn10] == lab[:, None]).mean())
        out["pairs"].append({
            "classes": [CLASSES[i], CLASSES[j]], "ids": [i, j], "test_confusions": n_conf,
            "sign_convention_agreement": sign_ok,
            "n_sv_pair": int(len(sv_idx)), "sv_frac_of_pair_train": float(len(sv_idx) / (len(A) + len(B))),
            "frac_sv_at_bound": float(bounded.mean()),
            "support_vectors": sv_show,
            "near_boundary_test": [{"test_idx": int(t), "label": int(yte[t]), "pred": int(pred[t]),
                                    "f": float(f[t])} for t in near],
            "pair_test_err_rate": float(err.sum() / m.sum()),
            "frac_pair_errors_inside_margin": float((np.abs(f[err]) < 1).mean()) if err.any() else float("nan"),
            "frac_pair_correct_inside_margin": float((np.abs(f[m & ~err]) < 1).mean()),
            "mean_abs_f_errors": float(np.abs(f[err]).mean()) if err.any() else float("nan"),
            "mean_abs_f_correct": float(np.abs(f[m & ~err]).mean()),
            "fisher_ratio": fisher, "knn10_pair_purity": purity})

    # (c) ---------------------------------------------------------------------
    Xtr_full = scaler.transform(ftr_all)
    rng = np.random.default_rng(cfg["seed"])
    sil_idx = rng.choice(len(Xv), min(len(Xv), 5000), replace=False)

    def fit_cluster(method, k):
        if method == "kmeans":
            return KMeans(k, n_init=10, random_state=0).fit(Xtr_full)
        return GaussianMixture(k, covariance_type="diag", n_init=3, random_state=0, reg_covar=1e-4).fit(Xtr_full)

    csel = []
    for method in q["cluster_methods"]:
        for k in q["cluster_k"]:
            mdl = fit_cluster(method, k)
            lv = mdl.predict(Xv)
            sil = float(silhouette_score(Xv[sil_idx], lv[sil_idx])) if len(set(lv)) > 1 else -1.0
            rec = {"method": method, "k": k, "val_silhouette": sil,
                   "val_nmi_reference_only": float(normalized_mutual_info_score(yva, lv))}
            if method == "gmm":
                rec["val_bic"] = float(mdl.bic(Xv))
            csel.append(rec)
            print(f"[Q4c] {method} k={k} sil={sil:.3f}")
    out["cluster_selection"] = csel
    cb = max(csel, key=lambda r: r["val_silhouette"])
    out["cluster_selected"] = {"method": cb["method"], "k": cb["k"], "rule": "max validation silhouette (label-free)"}
    mdl = fit_cluster(cb["method"], cb["k"])
    lt = mdl.predict(Xt)                     # assignment uses no labels
    ltr = mdl.predict(Xtr_full)
    K = cb["k"]
    cont = np.zeros((K, 10), int)
    np.add.at(cont, (lt, yte), 1)             # labels used only from here on
    out["cluster_test"] = {"ARI": float(adjusted_rand_score(yte, lt)), "NMI": float(normalized_mutual_info_score(yte, lt)),
                           "contingency": cont.tolist()}
    # majority-class mapping learned from TRAIN labels, then applied to test clusters
    tr_cont = np.zeros((K, 10), int)
    np.add.at(tr_cont, (ltr, ytr_all), 1)
    mapping = tr_cont.argmax(1)
    out["cluster_test"]["majority_map_acc"] = float((mapping[lt] == yte).mean())
    out["cluster_test"]["purity"] = float(cont.max(1).sum() / cont.sum())

    # classification vs clustering disagreement ---------------------------------
    P = cont / cont.sum(0, keepdims=True)    # P[k, c] = p(cluster k | class c)
    co = P.T @ P                              # co[i,j] = prob. random i and j test points share a cluster
    rows = []
    for i in range(10):
        for j in range(i + 1, 10):
            svm_conf = (cm[i, j] + cm[j, i]) / (cm[i].sum() + cm[j].sum())
            rows.append({"pair": [CLASSES[i], CLASSES[j]], "ids": [i, j], "svm_confusion_rate": float(svm_conf),
                         "cluster_coassignment": float(co[i, j]),
                         "within_i": float(co[i, i]), "within_j": float(co[j, j])})
    out["pair_agreement"] = rows
    svm_c = np.array([r["svm_confusion_rate"] for r in rows])
    clu_c = np.array([r["cluster_coassignment"] for r in rows])
    # rank-normalised disagreement: clustering says "same group", SVM says "well separated"
    rk = lambda v: np.argsort(np.argsort(v)) / (len(v) - 1)
    dis = rk(clu_c) - rk(svm_c)
    out["max_disagreement_pair"] = rows[int(np.argmax(dis))]
    out["min_disagreement_pair"] = rows[int(np.argmin(dis))]
    from scipy.stats import spearmanr
    out["spearman_svm_vs_cluster_pairs"] = float(spearmanr(svm_c, clu_c).correlation)
    out["per_class"] = [{"class": CLASSES[c], "svm_recall": out["test_per_class_recall"][c],
                         "cluster_concentration": float(P[:, c].max()),
                         "dominant_cluster": int(P[:, c].argmax())} for c in range(10)]
    save_json(out, rdir(cfg, "q4.json"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    run(load_cfg(smoke=ap.parse_args().smoke))
