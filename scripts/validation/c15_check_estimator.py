"""Check 15 code check: the c15_score.py LeafletFA group estimator, run on the REAL TMS TRAIN counts with the production
seed-0 fit, must reproduce the saved 10.1 errors (c10/TMS_s0_accuracy_units.tsv.gz, err_leafletfa vs TEST pseudobulk)."""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
from c10_catalog import labels, onehot, tv, fit_dir, MIN_TRAIN  # noqa: E402
P = vlib.paths("TMS"); z = np.load(os.path.join(fit_dir("TMS", 0), "fit.npz"), allow_pickle=True)
inv = np.asarray(z["inv_ev"], np.int64); E_ = len(z["event_id"]); J = len(inv)
phi = z["phi"].astype(np.float64); psi = z["psi_mean"].astype(np.float64); gam = np.asarray(z["bb_conc_vec"], np.float64)
cells, t, _ = labels("TMS", P["train"]); assert np.array_equal(cells, z["cell_id"].astype(str))
Y = vlib.read_counts(P["train"]).tocsr().astype(np.float64)
Em = sp.csr_matrix((np.ones(J), (np.arange(J), inv)), shape=(J, E_)); N = (Y @ Em).tocsr()
Nc = N.tocoo(); V = sp.csr_matrix((Nc.data / (gam[Nc.col] + Nc.data), (Nc.row, Nc.col)), shape=N.shape)
Yc = Y.tocoo(); Ncj = np.asarray(N[Yc.row, inv[Yc.col]]).ravel()
R = sp.csr_matrix((Yc.data / (gam[inv[Yc.col]] + Ncj), (Yc.row, Yc.col)), shape=Y.shape)
cnt = pd.Series(t).value_counts(); types = sorted(cnt[cnt >= MIN_TRAIN].index); code = {x: i for i, x in enumerate(types)}
g = np.array([code.get(x, len(types)) for x in t]); G = len(types) + 1
O = onehot(g, G); out = (O @ phi) @ psi + np.asarray((O @ R).todense())
for k in range(G):
    i = np.flatnonzero(g == k); A = V[i].T @ phi[i]; out[k] -= np.einsum("jk,kj->j", A[inv], psi)
est = out / np.bincount(g, minlength=G)[:, None]
_, tt, _ = labels("TMS", P["test"]); Yt = vlib.read_counts(P["test"]).tocsr(); Nt = (Yt @ Em).tocsr()
gt = np.array([code.get(x, len(types)) for x in tt]); Ot = onehot(gt, G)
yt, nt = np.asarray((Ot @ Yt).todense()), np.asarray((Ot @ Nt).todense())
with np.errstate(invalid="ignore", divide="ignore"):
    tpsi = np.nan_to_num(yt / nt[:, inv])
err = tv(est, tpsi, inv, E_)
U = pd.read_csv(os.path.join(vlib.V, "c10", "TMS_s0_accuracy_units.tsv.gz"), sep="\t")
d = np.abs(err[U.cell_type.map(code).to_numpy(), U.atse.to_numpy()] - U.err_leafletfa.to_numpy())
print(f"units {len(U):,}; max |diff| {d.max():.2e}; mean |diff| {d.mean():.2e}")
assert d.max() < 1e-4
