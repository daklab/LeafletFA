"""Check 15, step 3 (CPU; amendment 10 + note 1): score the 3 simulation fits against truth.
    python c15_score.py <full|f30|f10>
Measure 1: per real type (halves pooled), null ATSEs (applied shift 0 in every type), error TV(estimate, truth) for
  LeafletFA (catalog estimator: mean rho~ over the type's cells, the sim fit's gamma_e), raw (pooled fallback) and
  comparator B (c from an inner 70/30 split of the simulated cells within type x mouse, seed 1, amendment 9 grid; truth
  never used). Bins: simulated reads at (t, e) and type size. Paired differences with the 10.1 bootstrap (types, 1,000,
  seed 0). Primary = own-truth pairs (note 1); all pairs secondary.
Measure 2 (LeafletFA and raw part; LeafCutter is c15_leafcutter.py): units (t, e) over scored ATSEs, types with >= 4 mice
  having cells in both halves; label = applied shift > 0. Score = TV(t|A, t|B). AUROC and precision at recall 0.5 overall,
  by delta (planted with that delta vs all nulls) and by simulated reads bin; CIs by bootstrap over types (1,000, seed 0).
Writes $V/c15/score/<D>_{m1.tsv,m2.tsv,m2_units.npz,inner_grid.tsv,summary.json}."""
import os, sys, json, time
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
from c10_catalog import labels, onehot, tv, MIN_TRAIN, MIN_TEST_CELLS, MIN_TEST_READS, MIN_TEST_COV  # noqa: E402

GRID = [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
D = sys.argv[1]; C15 = os.path.join(vlib.V, "c15"); OUT = os.path.join(C15, "score"); os.makedirs(OUT, exist_ok=True)
RB = [-0.5, 0.5, 19.5, 99.5, np.inf]; RL = ["0", "1-19", "20-99", ">=100"]
SB = [9, 49, 199, 999, np.inf]; SL = ["10-49", "50-199", "200-999", ">=1000"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


tr = np.load(os.path.join(C15, "truth.npz"), allow_pickle=True)
types = tr["types"].astype(str); T = len(types); g = tr["g"]; half_b = tr["half_b"]; inv = tr["inv"]
truth, truthB, shift, delta = tr["truth"], tr["truthB"], tr["shift"], tr["delta"]
scored_e, own = tr["scored_e"], tr["own"]; E_ = len(scored_e); J = len(inv)
cells = tr["cell_id"].astype(str)
_, t_lab, mouse = labels("TMS", vlib.paths("TMS")["train"])
assert np.array_equal(np.array([types[k] if k < T else None for k in g], dtype=object)[g < T], t_lab[g < T])
z = np.load(os.path.join(C15, f"fit_{D}", "fit", "fit.npz"), allow_pickle=True)
assert np.array_equal(z["cell_id"].astype(str), cells) and np.array_equal(np.asarray(z["inv_ev"]), inv)
phi = z["phi"].astype(np.float64); psi = z["psi_mean"].astype(np.float64); gam = np.asarray(z["bb_conc_vec"], np.float64)
Y = sp.load_npz(os.path.join(C15, f"sim_{D}_train.npz")).tocsr().astype(np.float64)
Em = sp.csr_matrix((np.ones(J), (np.arange(J), inv)), shape=(J, E_)); N = (Y @ Em).tocsr()
log(f"{D}: {Y.shape[0]:,} cells, {Y.sum():,.0f} reads, {T} types")
Nc = N.tocoo(); V = sp.csr_matrix((Nc.data / (gam[Nc.col] + Nc.data), (Nc.row, Nc.col)), shape=N.shape)
Yc = Y.tocoo(); Ncj = np.asarray(N[Yc.row, inv[Yc.col]]).ravel()
R = sp.csr_matrix((Yc.data / (gam[inv[Yc.col]] + Ncj), (Yc.row, Yc.col)), shape=Y.shape); del Yc, Ncj, Nc


def lfa_sums(rows, gg, G):
    """(G x J) sums over each group's cells of rho~ = (gamma rho + y) / (gamma + N); group cell counts"""
    O = onehot(gg, G); ph = phi[rows]; Vr = V[rows]
    out = (O @ ph) @ psi + np.asarray((O @ R[rows]).todense())
    for k in range(G):
        i = np.flatnonzero(gg == k)
        if len(i):
            A = (Vr[i].T @ ph[i])                                                  # ATSE x K
            out[k] -= np.einsum("jk,kj->j", A[inv], psi)
    return out, np.bincount(gg, minlength=G).astype(np.float64)


def raw_sums(rows, gg, G):
    O = onehot(gg, G)
    return np.asarray((O @ Y[rows]).todense()), np.asarray((O @ N[rows]).todense())


def with_fallback(y, n, p):
    nJ = n[:, inv]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(nJ > 0, y / np.maximum(nJ, 1e-300), p[None, :])


def est_B(y, n, p, c):
    nJ = n[:, inv]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(nJ + c > 0, (y + c * p[None, :]) / np.maximum(nJ + c, 1e-300), p[None, :])


allrows = np.arange(Y.shape[0]); G = T + 1
p_all = np.asarray(Y.sum(0)).ravel() / np.maximum(np.asarray(N.sum(0)).ravel()[inv], 1e-300)
# ---------------- comparator B: c on an inner 70/30 split of the simulated cells (truth not used) ----------------
rng = np.random.default_rng(1); inner_te = np.zeros(len(g), bool)
for _, idx in pd.DataFrame(dict(g=g, m=mouse)).groupby(["g", "m"]).indices.items():
    k = int(np.floor(0.3 * len(idx)))
    if k:
        inner_te[rng.permutation(idx)[:k]] = True
itr, ite = np.flatnonzero(~inner_te), np.flatnonzero(inner_te)
yi, ni = raw_sums(itr, g[itr], G)
p_in = np.asarray(Y[itr].sum(0)).ravel() / np.maximum(np.asarray(N[itr].sum(0)).ravel()[inv], 1e-300)
yt_, nt_ = raw_sums(ite, g[ite], G); cov_ = np.asarray((onehot(g[ite], G) @ (N[ite] > 0).astype(np.float64)).todense())
with np.errstate(invalid="ignore", divide="ignore"):
    tpsi_i = np.nan_to_num(yt_ / nt_[:, inv])
ok_i = (nt_ >= MIN_TEST_READS) & (cov_ >= MIN_TEST_COV)
ok_i &= ((np.bincount(g[ite], minlength=G) >= MIN_TEST_CELLS) & (np.bincount(g[itr], minlength=G) >= MIN_TRAIN))[:, None]
ok_i[-1] = False
grid = []
for c in GRID:
    grid.append(dict(depth=D, c=c, inner_units=int(ok_i.sum()), mean_err=float(tv(est_B(yi, ni, p_in, c), tpsi_i, inv, E_)[ok_i].mean())))
grid = pd.DataFrame(grid); grid.to_csv(os.path.join(OUT, f"{D}_inner_grid.tsv"), sep="\t", index=False)
c_star = float(grid.loc[grid.mean_err.idxmin(), "c"]); log(f"B: c = {c_star}"); del yi, ni, yt_, nt_, cov_, tpsi_i

# ---------------- Measure 1: per type, halves pooled, null ATSEs ----------------
Sl, nl = lfa_sums(allrows, g, G); yr, nr = raw_sums(allrows, g, G)
est = dict(leafletfa=Sl[:T] / nl[:T, None], raw=with_fallback(yr, nr, p_all)[:T], B=est_B(yr, nr, p_all, c_star)[:T])
reads = nr[:T]; ncell = nl[:T]; del Sl
null_e = scored_e & ~(shift > 0).any(0)
err = {k: tv(v, truth, inv, E_) for k, v in est.items()}
kk, ee = np.nonzero(np.broadcast_to(null_e[None, :], (T, E_)))
U1 = pd.DataFrame(dict(type=kk, atse=ee, own=own[kk, ee], reads=reads[kk, ee], cells=ncell[kk],
                       **{f"err_{k}": err[k][kk, ee] for k in err}))
U1["read_bin"] = pd.cut(U1.reads, RB, labels=RL); U1["size_bin"] = pd.cut(U1.cells, SB, labels=SL)
U1["d_raw"] = U1.err_leafletfa - U1.err_raw; U1["d_B"] = U1.err_leafletfa - U1.err_B


def boot_ratio(sub, col):
    rng_ = np.random.default_rng(0); per = sub.groupby("type")[col].agg(["sum", "size"]); s, n = per["sum"].to_numpy(), per["size"].to_numpy()
    idx = rng_.integers(0, len(per), (1000, len(per)))
    return np.quantile(s[idx].sum(1) / n[idx].sum(1), [0.025, 0.975])


rows = []
for subset, S_ in (("own", U1[U1.own]), ("all", U1)):
    for by, col in (("all", None), ("reads", "read_bin"), ("size", "size_bin")):
        for lev, sub in ([("all", S_)] if col is None else S_.groupby(col, observed=True)):
            r = dict(depth=D, pairs=subset, by=by, bin=str(lev), units=len(sub), types=sub.type.nunique(),
                     err_leafletfa=sub.err_leafletfa.mean(), err_raw=sub.err_raw.mean(), err_B=sub.err_B.mean())
            for d in ("d_raw", "d_B"):
                lo, hi = boot_ratio(sub, d); r.update({d: sub[d].mean(), f"{d}_lo": lo, f"{d}_hi": hi})
            rows.append(r)
M1 = pd.DataFrame(rows); M1.to_csv(os.path.join(OUT, f"{D}_m1.tsv"), sep="\t", index=False)
log("Measure 1 done"); print(M1.round(4).to_string(index=False), flush=True)

# ---------------- Measure 2: halves A vs B ----------------
hg = g * 2 + half_b.astype(int); G2 = 2 * G
Sh, nh = lfa_sums(allrows, hg, G2); yh, nhr = raw_sums(allrows, hg, G2)
lh = Sh / np.maximum(nh, 1)[:, None]; rh = with_fallback(yh, nhr, p_all); del Sh
s_lfa = tv(lh[0:2 * T:2], lh[1:2 * T:2], inv, E_); s_raw = tv(rh[0:2 * T:2], rh[1:2 * T:2], inv, E_)
mm = pd.DataFrame(dict(g=g, m=mouse, b=half_b))[g < T].groupby(["g", "m"]).b.agg(["min", "max"])
both = mm[(mm["min"] == 0) & (mm["max"] == 1)].reset_index().groupby("g").size().reindex(range(T), fill_value=0).to_numpy()
types2 = np.flatnonzero(both >= 4)
kk, ee = np.nonzero(np.broadcast_to(scored_e[None, :], (T, E_)) & np.isin(np.arange(T), types2)[:, None])
lab = shift[kk, ee] > 0
U2 = dict(type=kk, atse=ee, label=lab, delta=np.where(lab, delta[ee], 0.0), shift=shift[kk, ee], own=own[kk, ee],
          reads=reads[kk, ee], readsA=nhr[2 * kk, ee], readsB=nhr[2 * kk + 1, ee], s_lfa=s_lfa[kk, ee], s_raw=s_raw[kk, ee])
np.savez_compressed(os.path.join(OUT, f"{D}_m2_units.npz"), **U2)
log(f"Measure 2: {len(types2)} of {T} types with >= 4 mice in both halves; {len(kk):,} units, {lab.sum():,} planted")


def auc_prec(iu, nu, y, w):
    """weighted AUROC (ties half) and precision at recall 0.5; iu = rank of each unit's score among nu unique values;
    w = per-unit multiplicity"""
    wp = np.bincount(iu, weights=w * y, minlength=nu); wn = np.bincount(iu, weights=w * ~y, minlength=nu)
    Wp, Wn = wp.sum(), wn.sum()
    if Wp == 0 or Wn == 0:
        return np.nan, np.nan
    cn = np.cumsum(wn) - wn                                                        # negatives strictly below
    auc = (wp * (cn + 0.5 * wn)).sum() / (Wp * Wn)
    cp_d, ca_d = np.cumsum(wp[::-1]), np.cumsum((wp + wn)[::-1])                  # descending score
    i = np.searchsorted(cp_d, 0.5 * Wp)
    return auc, cp_d[i] / ca_d[i]


def m2_row(mask, name, by, lev):
    tp = U2["type"][mask]; yv = U2["label"][mask]; ut, it = np.unique(tp, return_inverse=True)
    r = dict(depth=D, by=by, bin=lev, units=int(mask.sum()), planted=int(yv.sum()), types=len(ut))
    rng_ = np.random.default_rng(0); W = np.stack([np.bincount(rng_.integers(0, len(ut), len(ut)), minlength=len(ut))
                                                    for _ in range(1000)])
    for s in ("s_lfa", "s_raw"):
        u, iu = np.unique(U2[s][mask], return_inverse=True); a, p = auc_prec(iu, len(u), yv, np.ones(len(iu)))
        bs = np.array([auc_prec(iu, len(u), yv, W[b][it].astype(np.float64)) for b in range(1000)])
        r.update({f"auroc_{s[2:]}": a, f"auroc_{s[2:]}_lo": np.nanquantile(bs[:, 0], .025), f"auroc_{s[2:]}_hi": np.nanquantile(bs[:, 0], .975),
                  f"prec50_{s[2:]}": p, f"prec50_{s[2:]}_lo": np.nanquantile(bs[:, 1], .025), f"prec50_{s[2:]}_hi": np.nanquantile(bs[:, 1], .975)})
    log(f"  m2 {by} {lev}: AUROC lfa {r['auroc_lfa']:.3f} raw {r['auroc_raw']:.3f}")
    return r


allm = np.ones(len(kk), bool); rb = pd.cut(U2["reads"], RB, labels=RL).astype(str)
rows = [m2_row(allm, "all", "all", "all")]
for dv in (0.1, 0.3, 0.5):
    rows.append(m2_row(~U2["label"] | np.isclose(U2["delta"], dv), "delta", "delta", str(dv)))
for lev in RL:
    rows.append(m2_row(rb == lev, "reads", "reads", lev))
rows.append(m2_row(U2["own"], "own", "own_truth", "own"))
M2 = pd.DataFrame(rows); M2.to_csv(os.path.join(OUT, f"{D}_m2.tsv"), sep="\t", index=False)
o = M1[(M1.pairs == "own") & (M1.by == "all")].iloc[0]; a = M1[(M1.pairs == "all") & (M1.by == "all")].iloc[0]
summ = dict(depth=D, reads=float(Y.sum()), c_B=c_star, c_at_grid_end=c_star in (GRID[0], GRID[-1]),
            null_scored_atses=int(null_e.sum()), m1_own=o.to_dict(), m1_all=a.to_dict(),
            m2_types=int(len(types2)), m2_all=M2.iloc[0].to_dict())
json.dump(summ, open(os.path.join(OUT, f"{D}_summary.json"), "w"), indent=1, default=float)
print(M2.round(3).to_string(index=False)); log("done")
