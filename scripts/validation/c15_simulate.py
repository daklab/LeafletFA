"""Check 15, step 1 (CPU; amendment 10): semi-synthetic TMS TRAIN with truth cell-type psi from real pseudobulk, gamma_e by
method of moments, planted A/B shifts in 30 % of scored ATSEs, three depths (f = 1, 0.3, 0.1).
    python c15_simulate.py   -> $V/c15/{sim_full,sim_f30,sim_f10}_train.npz, cells.tsv, var.tsv, truth.npz, sim_summary.json
"""
import os, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
from c10_catalog import labels, MIN_TRAIN  # noqa: E402

OUT = os.path.join(vlib.V, "c15"); os.makedirs(OUT, exist_ok=True)
P = vlib.paths("TMS")
jid, ev = vlib.read_var(P["train"]); _, inv = vlib.event_index(ev); J = len(inv); E_ = inv.max() + 1
cells, t, mouse = labels("TMS", P["train"])
assert np.array_equal(vlib.read_cells(P["train"]).astype(str), cells)
Y = vlib.read_counts(P["train"]).tocsr().astype(np.float64)
Em = sp.csr_matrix((np.ones(J), (np.arange(J), inv)), shape=(J, E_))
N = (Y @ Em).tocsr()
cnt = pd.Series(t).value_counts(); types = sorted(cnt[cnt >= MIN_TRAIN].index); T = len(types)
code = {x: i for i, x in enumerate(types)}; g = np.array([code.get(x, T) for x in t])        # T = unscored cells
O = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(T + 1, len(g)))
yt = np.asarray((O @ Y).todense())[:T]; nt = np.asarray((O @ N).todense())[:T]
cov = np.asarray((O @ (N > 0).astype(np.float64)).todense())[:T]
pool = np.asarray(Y.sum(0)).ravel() / np.maximum(np.asarray(N.sum(0)).ravel()[inv], 1e-300)
well = (nt >= 100) & (cov >= 10)                                                            # T x E
scored_e = well.any(0)
# ---- truth psi: own pseudobulk where well covered; copied from a random well-covered type at e elsewhere ----
rng = np.random.default_rng(11)
with np.errstate(invalid="ignore", divide="ignore"):
    raw = yt / nt[:, inv]
truth = np.repeat(pool[None, :], T, 0); own = np.zeros((T, E_), bool); donor = np.full((T, E_), -1)
jidx = [np.flatnonzero(inv == e) for e in range(E_)]
for e in np.flatnonzero(scored_e):
    wt = np.flatnonzero(well[:, e]); js = jidx[e]
    for k in range(T):
        d = k if well[k, e] else wt[rng.integers(len(wt))]
        truth[k, js] = raw[d, js]; donor[k, e] = d; own[k, e] = well[k, e]
# ---- gamma_e, method of moments on real counts (cells of well-covered types, N >= 2); vectorized ----
# sum_j (y - N psi)^2 / psi = sum_j y^2 / psi - N^2 over psi > 0 junctions (own truth = the pseudobulk, so y > 0 only where psi > 0)
N.sort_indices(); Nc = N.tocoo(); keyN = Nc.row.astype(np.int64) * E_ + Nc.col
Yc = Y.tocoo(); rows, cols, y = Yc.row, Yc.col, Yc.data; ce = inv[cols]
idx = np.searchsorted(keyN, rows.astype(np.int64) * E_ + ce); assert (keyN[idx] == rows.astype(np.int64) * E_ + ce).all()
cg = g[rows]; ok = cg < T
ok[ok] = well[cg[ok], ce[ok]]
psi_e = np.where(ok, truth[np.minimum(cg, T - 1), cols], 0.0); ok &= psi_e > 0
Ssum = np.bincount(idx[ok], weights=y[ok] ** 2 / psi_e[ok], minlength=len(Nc.data))
gN = g[Nc.row]; m = (gN < T) & (Nc.data >= 2)
m[m] = well[gN[m], Nc.col[m]]
mp = np.asarray(((truth > 0).astype(np.float64) @ Em))                                     # T x E: junctions with psi > 0
n2 = Nc.data[m]; e2 = Nc.col[m]; mm = mp[gN[m], e2]
Sc = Ssum[m] - n2 ** 2
X = np.bincount(e2, weights=Sc, minlength=E_); S1 = np.bincount(e2, weights=(mm - 1) * n2, minlength=E_)
S2 = np.bincount(e2, weights=(mm - 1) * n2 ** 2, minlength=E_)
del Yc, rows, cols, y, ce, idx, cg, ok, psi_e, Ssum
with np.errstate(invalid="ignore", divide="ignore"):
    gam = np.clip((S2 - X) / (X - S1), 0.5, 1e4)
gam[(X <= S1) & (S1 > 0)] = 1e4
has = (S1 > 0) & np.isfinite(gam)
gam[~has] = np.median(gam[has])
# ---- halves and planted shifts ----
rng = np.random.default_rng(10); half_b = np.zeros(len(g), bool)
for _, idx in pd.DataFrame(dict(g=g, m=mouse)).groupby(["g", "m"]).indices.items():
    half_b[rng.permutation(idx)[: len(idx) // 2]] = True
rng = np.random.default_rng(12); se = np.flatnonzero(scored_e)
planted = np.zeros(E_, bool); pl = rng.permutation(se)[: int(round(0.3 * len(se)))]; planted[pl] = True
delta = np.zeros(E_); delta[pl] = rng.permutation(np.resize([0.1, 0.3, 0.5], len(pl)))
truthB = truth.copy(); shift = np.zeros((T, E_))
for e in pl:
    js = jidx[e]; jt = js[np.argmin(pool[js])]
    for k in range(T):
        p = truth[k, js]; cand = [i for i in range(len(js)) if js[i] != jt]
        jf = js[cand[int(np.argmax(p[cand]))]]
        s = min(delta[e], truth[k, jf]); truthB[k, jf] -= s; truthB[k, jt] += s; shift[k, e] = s
# ---- groups: type x half (+ unscored A/B) ----
grp = g * 2 + half_b.astype(int); G2 = 2 * (T + 1)
psi_g = np.vstack([np.vstack([truth[k], truthB[k]]) if k < T else np.vstack([pool, pool]) for k in range(T + 1)])
phi = np.zeros((len(g), G2)); phi[np.arange(len(g)), grp] = 1.0
pd.DataFrame({"cell_id": cells}).to_csv(os.path.join(OUT, "cells.tsv"), sep="\t", index=False)
pd.DataFrame({"junction_id": jid, "event_id": ev}).to_csv(os.path.join(OUT, "var.tsv"), sep="\t", index=False)
np.savez_compressed(os.path.join(OUT, "truth.npz"), types=np.array(types), g=g, half_b=half_b, truth=truth, truthB=truthB,
                    shift=shift, planted=planted, delta=delta, scored_e=scored_e, well=well, own=own, donor=donor, gamma=gam,
                    pool=pool, inv=inv, cell_id=cells)
summ = dict(cells=int(len(g)), types=T, atses=int(E_), scored_atses=int(scored_e.sum()), planted=int(planted.sum()),
            well_pairs=int(well.sum()), own_share_scored=float(own[:, scored_e].mean()),
            gamma_median=float(np.median(gam)), gamma_q10_q90=[float(np.quantile(gam, .1)), float(np.quantile(gam, .9))],
            gamma_at_floor_0p5=float((gam <= 0.5).mean()), gamma_at_cap=float((gam >= 1e4).mean()))
fit_g = np.load(os.path.join(vlib.B, "TMS", "full_K20_s0_f64", "fit", "fit.npz"))["bb_conc_vec"]
summ["production_gamma_median"] = float(np.median(fit_g)); summ["production_gamma_at_2"] = float((fit_g <= 2.0001).mean())
for name, f, seed in (("full", None, 21), ("f30", 0.3, 22), ("f10", 0.1, 23)):
    S = vlib.simulate_fast(Y, inv, phi, psi_g, gam, thin=f, seed=seed)
    sp.save_npz(os.path.join(OUT, f"sim_{name}_train.npz"), S)
    summ[f"reads_{name}"] = float(S.sum()); print(name, f"{S.sum():,.0f} reads", flush=True)
summ["reads_real"] = float(Y.sum())
json.dump(summ, open(os.path.join(OUT, "sim_summary.json"), "w"), indent=1); print(json.dumps(summ, indent=1))
