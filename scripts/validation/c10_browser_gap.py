"""Read-only diagnosis (Karin 2026-11-06): why browser rankings (LeafletFA delta psi) disagree with raw TRAIN pseudobulk delta psi.
    python c10_browser_gap.py TMS "Brain_Non-Myeloid_NEURON" "Brain_Non-Myeloid_ASTROCYTE"
Checks (1) bug: the saved estimates and the portal's raw table reproduce from the fit and counts; (2) decomposition of the gap at the lead
junction of the top 100 ATSEs (browser default filters: >= 20 reads and >= 3 cells with reads in both types) into
  uncovered = [LeafletFA all cells] - [LeafletFA covered cells]   (cells with no reads at the ATSE contribute their program prior phi_c psi)
  shrink    = [LeafletFA covered cells] - [raw mean of y/N over covered cells]   (per-cell update (1 - V) phi psi + R, V = N / (gamma + N))
  weight    = [raw cell mean] - [raw read-pooled sum y / sum N]   (the browser's raw psi is read-pooled)
Writes c10_browser_gap/<ds>_<a>_vs_<b>.tsv and prints the summary."""
import os, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
from c10_catalog import Data, Groups, type_setup, estimates, log  # noqa: E402
ds, ta, tb = sys.argv[1:4]; OUT = os.path.join(vlib.V, "c10_browser_gap"); os.makedirs(OUT, exist_ok=True)
D = Data(ds, 0); types, g = type_setup(D); G = len(types) + 1
E = np.load(os.path.join(vlib.V, "c10", f"{ds}_s0_estimates.npz"), allow_pickle=True); assert list(E["types"].astype(str)) == list(types)
p, *_ = estimates(D, Groups(D, np.arange(len(g)), g, G), draws=False)
print("bug check 1: max |recomputed point - saved point|", float(np.abs(p[:-1] - E["point"]).max()))
Tb = np.load(os.path.join(vlib.V, "portal_browser", "precomputed", f"{ds}_train_tables.npz"))
psi, phi = D.post["psi_mean"], D.post["phi"]
def parts(k):
    i = np.flatnonzero(g == k); Y = D.Y[i].toarray().astype(np.float64); N = D.Ncell[i].toarray()[:, D.inv].astype(np.float64)
    V = D.V[i].toarray()[:, D.inv]; R = D.R[i].toarray(); prior = np.asarray(phi[i]) @ np.asarray(psi)
    rho = (1 - V) * prior + R; cov = N > 0; nc = cov.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = dict(lfa_all=rho.mean(0), lfa_cov=np.where(cov, rho, 0).sum(0) / nc, lfa_unc=np.where(~cov, prior, 0).sum(0) / (len(i) - nc),
                 raw_cell=np.where(cov, Y / np.where(cov, N, 1), 0).sum(0) / nc, raw_read=Y.sum(0) / N.sum(0), frac_cov=nc / len(i),
                 v_cov=np.where(cov, V, 0).sum(0) / nc, jreads=Y.sum(0))
    assert np.allclose(d["lfa_all"], E["point"][k], atol=1e-4), "per-cell rebuild does not give the saved point"
    assert np.array_equal(d["jreads"], Tb["jreads"][k].astype(np.float64)), "portal raw reads differ"
    return d
ka, kb = types.index(ta), types.index(tb); A, B = parts(ka), parts(kb); print("bug check 2: per-cell rebuild = saved point; portal junction reads = counts (asserted)")
dP = E["point"][ka] - E["point"][kb]; ok = (E["reads"][ka] >= 20) & (E["cov"][ka] >= 3) & (E["reads"][kb] >= 20) & (E["cov"][kb] >= 3)
best = pd.DataFrame(dict(a=D.inv, d=np.abs(dP), j=np.arange(D.J)))
best = best[ok[D.inv]].sort_values("d", ascending=False).drop_duplicates("a").head(100)
j = best.j.values; X = pd.DataFrame(dict(atse=D.post["event_id"][D.inv[j]], junction=D.post["junction_id"][j], lfa_dpsi=dP[j]))
for k in ("lfa_all", "lfa_cov", "lfa_unc", "raw_cell", "raw_read", "frac_cov", "v_cov"):
    X[k + "_A"] = A[k][j]; X[k + "_B"] = B[k][j]
X["raw_dpsi"] = X.raw_read_A - X.raw_read_B; X["gap"] = X.lfa_dpsi - X.raw_dpsi
X["uncovered"] = (X.lfa_all_A - X.lfa_cov_A) - (X.lfa_all_B - X.lfa_cov_B)
X["shrink"] = (X.lfa_cov_A - X.raw_cell_A) - (X.lfa_cov_B - X.raw_cell_B)
X["weight"] = (X.raw_cell_A - X.raw_read_A) - (X.raw_cell_B - X.raw_read_B)
assert np.allclose(X.gap, X.uncovered + X.shrink + X.weight, atol=1e-4)
X["dominant"] = X[["uncovered", "shrink", "weight"]].abs().idxmax(axis=1); X["sign_flip"] = np.sign(X.lfa_dpsi) != np.sign(X.raw_dpsi)
X.to_csv(os.path.join(OUT, f"{ds}_{ta}_vs_{tb}.tsv".replace("/", "-").replace(" ", "_")), sep="\t", index=False, float_format="%.4g")
big = X.gap.abs() > 0.3
S = dict(dataset=ds, a=ta, b=tb, cells_a=int((g == ka).sum()), cells_b=int((g == kb).sum()), median_abs_gap=float(X.gap.abs().median()), gap_gt_0_3=int(big.sum()),
         sign_flips=int(X.sign_flip.sum()), median_abs=dict(uncovered=float(X.uncovered.abs().median()), shrink=float(X.shrink.abs().median()), weight=float(X.weight.abs().median())),
         dominant_all=X.dominant.value_counts().to_dict(), dominant_gap_gt_0_3=X.dominant[big].value_counts().to_dict(),
         median_frac_cov=dict(A=float(X.frac_cov_A.median()), B=float(X.frac_cov_B.median())), median_v_cov=dict(A=float(X.v_cov_A.median()), B=float(X.v_cov_B.median())))
json.dump(S, open(os.path.join(OUT, f"{ds}_summary_{ka}_{kb}.json"), "w"), indent=1); print(json.dumps(S, indent=1))
print(X.sort_values("gap", key=abs, ascending=False).head(8)[["atse", "lfa_dpsi", "raw_dpsi", "uncovered", "shrink", "weight", "frac_cov_A", "frac_cov_B"]].round(3).to_string())
