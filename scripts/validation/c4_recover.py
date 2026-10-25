"""Check 4, step 3 (CPU): how well a refit on simulated TMS data recovers the true psi and phi.

    python c4_recover.py

Truth: check4/truth.npz (the seed-0 TMS fit). Fits: check4/fit_sim{A,B}/fit/fit.npz (A = TMS depth,
B = EasySci-like, thinned 0.011). Programs are matched truth -> fit by the Hungarian algorithm on
centred-psi cosine (as check 3); recovery is scored on truth-live programs (argmax >= 1 % of cells):
  - psi: Pearson r of matched psi rows (raw) and centred cosine
  - phi: per-cell Pearson r between true phi and fitted phi with columns permuted by the matching
  - kNN (tissue | broad type, k 15, mouse-grouped 4-fold) on centred log phi: true vs fitted
Pre-registered pass (A): median matched psi r >= 0.9 and median per-cell phi r >= 0.7; B descriptive.
Writes tsv/c4_recovery.tsv and tsv/c4_recovery_programs.tsv.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402


def centred(psi, pi):
    pi = np.asarray(pi, np.float64); pi = pi / pi.sum()
    C = psi - pi @ psi
    return C / np.linalg.norm(C, axis=1, keepdims=True).clip(1e-12)


def knn(phi, lab, mouse, k=15):
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    X = np.log(np.clip(phi, 1e-6, None)); X -= X.mean(1, keepdims=True)
    pred = np.empty(len(lab), dtype=object)
    for tr, te in GroupKFold(4).split(X, lab, mouse):
        sc = StandardScaler().fit(X[tr])
        pred[te] = KNeighborsClassifier(k).fit(sc.transform(X[tr]), lab[tr]).predict(sc.transform(X[te]))
    return float((pred == lab).mean())


def main():
    C4 = os.path.join(vlib.V, "check4")
    t = np.load(os.path.join(C4, "truth.npz"), allow_pickle=True)
    psi_t, phi_t, pi_t = t["psi"].astype(np.float64), t["phi"].astype(np.float64), t["pi"]
    cells, lab, mouse = vlib.obs_labels(vlib.paths("TMS")["train"])
    assert np.array_equal(cells, t["cell_id"].astype(str))
    K = psi_t.shape[0]
    live = np.bincount(phi_t.argmax(1), minlength=K) / len(phi_t) >= 0.01
    knn_true = knn(phi_t, lab, mouse)
    rows, prog = [], []
    for name, depth in (("A", "TMS depth"), ("B", "EasySci-like (x0.011)")):
        p = os.path.join(C4, f"fit_sim{name}", "fit", "fit.npz")
        if not os.path.exists(p):
            print(f"sim {name}: fit missing"); continue
        f = np.load(p, allow_pickle=True)
        assert np.array_equal(f["cell_id"].astype(str), cells)
        psi_f, phi_f = f["psi_mean"].astype(np.float64), f["phi"].astype(np.float64)
        R = centred(psi_t, pi_t) @ centred(psi_f, f["pi"]).T
        i, j = linear_sum_assignment(-R)
        perm = np.empty(K, int); perm[i] = j
        r_psi = np.array([np.corrcoef(psi_t[k], psi_f[perm[k]])[0, 1] for k in range(K)])
        cos = R[np.arange(K), perm]
        phi_m = phi_f[:, perm]
        A = phi_t - phi_t.mean(1, keepdims=True); B = phi_m - phi_m.mean(1, keepdims=True)
        r_cell = (A * B).sum(1) / np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1)).clip(1e-12)
        kf = knn(phi_f, lab, mouse)
        rows.append(dict(sim=name, depth=depth, reads=int(f["losses"].size) and None,
                         live_true=int(live.sum()), median_psi_r_live=float(np.median(r_psi[live])),
                         median_centred_cos_live=float(np.median(cos[live])),
                         n_live_centred_cos_ge_0_8=int((cos[live] >= 0.8).sum()),
                         median_cell_phi_r=float(np.nanmedian(r_cell)),
                         knn_true_phi=knn_true, knn_fitted_phi=kf,
                         passed=(bool(np.median(r_psi[live]) >= 0.9 and np.nanmedian(r_cell) >= 0.7) if name == "A" else None)))
        for k in np.flatnonzero(live):
            prog.append(dict(sim=name, true_program=k + 1, matched_fit_program=perm[k] + 1, pi_true=float(np.asarray(pi_t)[k]),
                             psi_r=r_psi[k], centred_cos=cos[k]))
    out = os.path.join(vlib.V, "tsv")
    r = pd.DataFrame(rows).drop(columns="reads")
    r.to_csv(os.path.join(out, "c4_recovery.tsv"), sep="\t", index=False)
    pd.DataFrame(prog).to_csv(os.path.join(out, "c4_recovery_programs.tsv"), sep="\t", index=False)
    print(r.round(3).to_string(index=False))
    print(pd.DataFrame(prog).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
