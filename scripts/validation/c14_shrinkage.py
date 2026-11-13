"""Check 14 (amendment 9, Karin 2026-11-13): does LeafletFA's 10.1 gain over raw exceed simple shrinkage?
    python c14_shrinkage.py <TMS|Allen>
Read-only: seed-0 data split (TRAIN/TEST counts), the saved 10.1 units (c10/<ds>_s0_accuracy_units.tsv.gz). No posterior.
  A: psi = (y + 1/m_e) / (n + 1)                              (one read spread evenly over the ATSE's junctions)
  B: psi = (y + c p) / (n + c), p = pooled TRAIN psi (all TRAIN cells); c per atlas from an inner 70/30 TRAIN split
     within cell type x mouse (seed 1), grid fixed in the amendment; TEST is never used to choose c.
Same units, TV error vs TEST pseudobulk, bins and bootstrap (cell types, 1,000, seed 0) as 10.1.
Writes $V/c14_shrinkage/<ds>_{units.tsv.gz,inner_grid.tsv,accuracy.tsv,summary.json}."""
import os, sys, json, time
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
from c10_catalog import labels, fit_dir, onehot, tv, MIN_TRAIN, MIN_TEST_CELLS, MIN_TEST_READS, MIN_TEST_COV  # noqa: E402

GRID = [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
ds = sys.argv[1]; OUT = os.path.join(vlib.V, "c14_shrinkage"); os.makedirs(OUT, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


P = vlib.paths(ds)
z = np.load(os.path.join(fit_dir(ds, 0), "fit.npz"), allow_pickle=True)
inv = np.asarray(z["inv_ev"], np.int64); n_atse = len(z["event_id"]); J = len(inv)
m_e = np.bincount(inv, minlength=n_atse).astype(np.float64)[inv]                 # junctions in each junction's ATSE
E = sp.csr_matrix((np.ones(J), (np.arange(J), inv)), shape=(J, n_atse))
cells, t, mouse = labels(ds, P["train"]); assert np.array_equal(cells, np.asarray(z["cell_id"]).astype(str))
Y = vlib.read_counts(P["train"]).tocsr(); N = (Y @ E).tocsr()
_, tt, _ = labels(ds, P["test"]); Yt = vlib.read_counts(P["test"]).tocsr(); Nt = (Yt @ E).tocsr()
assert Y.shape[1] == J and Yt.shape[1] == J
log(f"{ds}: {Y.shape[0]:,} TRAIN, {Yt.shape[0]:,} TEST cells, {J:,} junctions")
cnt = pd.Series(t).value_counts(); types = sorted(cnt[cnt >= MIN_TRAIN].index)       # as c10 type_setup
code = {x: i for i, x in enumerate(types)}; G = len(types) + 1


def pb(Ym, Nm, g, G_):
    O = onehot(g, G_)
    return np.asarray((O @ Ym).todense()), np.asarray((O @ Nm).todense())       # G x J junction reads, G x ATSE reads


def est_B(y, n_atsewise, p, c):
    n = n_atsewise[:, inv]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n + c > 0, (y + c * p[None, :]) / np.maximum(n + c, 1e-300), p[None, :])


def units(te_g, te_Y, te_N, G_, tr_n_cells, min_train):
    """10.1 unit rule on a test part: types with >= min_train train and >= 5 test cells; >= 20 reads, >= 3 covered cells"""
    O = onehot(te_g, G_); nte = np.bincount(te_g, minlength=G_)
    yt, nt = np.asarray((O @ te_Y).todense()), np.asarray((O @ te_N).todense())
    cov = np.asarray((O @ (te_N > 0).astype(np.float64)).todense())
    with np.errstate(invalid="ignore", divide="ignore"):
        tpsi = np.nan_to_num(yt / nt[:, inv])
    ok = (nt >= MIN_TEST_READS) & (cov >= MIN_TEST_COV)
    ok &= ((nte >= MIN_TEST_CELLS) & (tr_n_cells >= min_train))[:, None]
    ok[-1] = False                                                                # last group = all other cells
    return tpsi, ok


# ---------------- choose c on an inner split of TRAIN (TEST untouched) ----------------
rng = np.random.default_rng(1); inner_te = np.zeros(len(t), bool)
for (_, _), idx in pd.DataFrame(dict(t=t, m=mouse)).groupby(["t", "m"]).indices.items():
    k = int(np.floor(0.3 * len(idx)))
    if k:
        inner_te[rng.permutation(idx)[:k]] = True
g_all = np.array([code.get(x, G - 1) for x in t])
itr, ite = np.flatnonzero(~inner_te), np.flatnonzero(inner_te)
yi, ni = pb(Y[itr], N[itr], g_all[itr], G)
p_in = np.asarray(Y[itr].sum(0)).ravel() / np.maximum(np.asarray(N[itr].sum(0)).ravel()[inv], 1e-300)
tpsi_i, ok_i = units(g_all[ite], Y[ite], N[ite], G, np.bincount(g_all[itr], minlength=G), MIN_TRAIN)
grid = []
for c in GRID:
    err = tv(est_B(yi, ni, p_in, c), tpsi_i, inv, n_atse)
    grid.append(dict(dataset=ds, c=c, inner_units=int(ok_i.sum()), mean_err=float(err[ok_i].mean())))
    log(f"inner c={c}: {grid[-1]['mean_err']:.5f}")
grid = pd.DataFrame(grid); grid.to_csv(os.path.join(OUT, f"{ds}_inner_grid.tsv"), sep="\t", index=False)
c_star = float(grid.loc[grid.mean_err.idxmin(), "c"]); at_end = c_star in (GRID[0], GRID[-1])
log(f"chosen c = {c_star}{' (grid end)' if at_end else ''}")

# ---------------- full TRAIN estimates and 10.1 units ----------------
y, n = pb(Y, N, g_all, G)
p = np.asarray(Y.sum(0)).ravel() / np.maximum(np.asarray(N.sum(0)).ravel()[inv], 1e-300)
nJ = n[:, inv]
with np.errstate(invalid="ignore", divide="ignore"):
    raw = np.where(nJ > 0, y / np.maximum(nJ, 1e-300), p[None, :])
    estA = (y + 1.0 / m_e[None, :]) / (nJ + 1.0)
estB = est_B(y, n, p, c_star)
gt = np.array([code.get(x, G - 1) for x in tt]); Ot = onehot(gt, G)
yt, nt = np.asarray((Ot @ Yt).todense()), np.asarray((Ot @ Nt).todense())
with np.errstate(invalid="ignore", divide="ignore"):
    tpsi = np.nan_to_num(yt / nt[:, inv])
err = {k: tv(v, tpsi, inv, n_atse) for k, v in (("raw", raw), ("A", estA), ("B", estB))}
U = pd.read_csv(os.path.join(vlib.V, "c10", f"{ds}_s0_accuracy_units.tsv.gz"), sep="\t")
k_ = U.cell_type.map(code).to_numpy(); e_ = U.atse.to_numpy()
for k in ("raw", "A", "B"):
    U[f"re_{k}"] = err[k][k_, e_]
dmax = float(np.abs(U.re_raw - U.err_raw).max()); log(f"raw error recompute vs saved 10.1: max |diff| = {dmax:.2e}")
assert dmax < 1e-6, "alignment check failed (amendment 9): stop"
U["err_A"], U["err_B"] = U.re_A, U.re_B
U["d_A"], U["d_B"], U["d_raw"] = U.err_leafletfa - U.err_A, U.err_leafletfa - U.err_B, U.err_leafletfa - U.err_raw
U.drop(columns=["re_raw", "re_A", "re_B"]).to_csv(os.path.join(OUT, f"{ds}_units.tsv.gz"), sep="\t", index=False)


def boot(sub, col):
    rng_ = np.random.default_rng(0)
    per = sub.groupby("cell_type")[col].agg(["sum", "size"])
    v = []
    for _ in range(1000):
        x = per.iloc[rng_.integers(0, len(per), len(per))]; v.append(x["sum"].sum() / x["size"].sum())
    return np.quantile(v, [0.025, 0.975])


rows = []
for kind, col in (("all", None), ("size", "size_bin"), ("reads", "read_bin")):
    for lev, sub in ([("all", U)] if col is None else U.groupby(col, observed=True)):
        r = dict(dataset=ds, by=kind, bin=str(lev), units=len(sub), types=sub.cell_type.nunique(),
                 err_leafletfa=sub.err_leafletfa.mean(), err_raw=sub.err_raw.mean(), err_A=sub.err_A.mean(), err_B=sub.err_B.mean())
        for d in ("d_A", "d_B", "d_raw"):
            lo, hi = boot(sub, d); r.update({d: sub[d].mean(), f"{d}_lo": lo, f"{d}_hi": hi})
        rows.append(r)
A = pd.DataFrame(rows); A.to_csv(os.path.join(OUT, f"{ds}_accuracy.tsv"), sep="\t", index=False)
o = A[A.by == "all"].iloc[0]
summ = dict(dataset=ds, c=c_star, c_at_grid_end=at_end, units=int(o.units), types=int(o.types), alignment_max_diff=dmax,
            leafletfa_minus_B=[float(o.d_B), float(o.d_B_lo), float(o.d_B_hi)],
            leafletfa_minus_A=[float(o.d_A), float(o.d_A_lo), float(o.d_A_hi)],
            leafletfa_minus_raw=[float(o.d_raw), float(o.d_raw_lo), float(o.d_raw_hi)],
            B_below_0_ci_excludes_0=bool(o.d_B_hi < 0))
json.dump(summ, open(os.path.join(OUT, f"{ds}_summary.json"), "w"), indent=1)
log(json.dumps(summ)); print(A.round(5).to_string(index=False))
