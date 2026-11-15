"""Check 15b (amendment 11) check before scoring: best single-program AUROC for half B over scored cells,
max_k max(AUC_k, 1 - AUC_k), per arm and depth (acc arm has no halves in the data: reference). Also amendment 10 fits.
    python c15b_halfcheck.py   -> $V/c15b/halfcheck.tsv"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
rows = []
for arm, root in (("a10", os.path.join(vlib.V, "c15")), ("acc", os.path.join(vlib.V, "c15b", "acc")), ("diff", os.path.join(vlib.V, "c15b", "diff"))):
    tr = np.load(os.path.join(root, "truth.npz"), allow_pickle=True); m = tr["g"] < len(tr["types"]); hb = tr["half_b"][m]
    for D in ("full", "f30", "f10"):
        f = os.path.join(root, f"fit_{D}", "fit", "fit.npz")
        if not os.path.exists(f):
            continue
        phi = np.load(f)["phi"][m]; a = np.array([roc_auc_score(hb, phi[:, k]) for k in range(phi.shape[1])]); s = np.maximum(a, 1 - a)
        k = int(np.argmax(s)); d = phi[hb].mean(0) - phi[~hb].mean(0)
        rows.append(dict(arm=arm, depth=D, best_program=k, best_auroc=float(s[k]), second_auroc=float(np.sort(s)[-2]),
                         max_abs_mean_phi_diff=float(np.abs(d).max()),
                         reading="none" if s[k] <= 0.60 else ("partial" if s[k] <= 0.75 else "confounded")))
R = pd.DataFrame(rows); R.to_csv(os.path.join(vlib.V, "c15b", "halfcheck.tsv"), sep="\t", index=False); print(R.round(3).to_string(index=False))
