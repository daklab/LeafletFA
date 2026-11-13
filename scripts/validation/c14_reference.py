"""Amendment 9 note 1 (Karin 2026-11-13): reference levels for the 10.1 errors, not tests; plus the display numbers for
the alternative panel b (figAN_b_abs): absolute errors of LeafletFA and raw with the 10.1 bootstrap, and the share of
catalog rows per bin.
    python c14_reference.py <TMS|Allen>
  upper  = TV(pooled TRAIN psi over all TRAIN cells, TEST pseudobulk) per 10.1 unit
  floor  = mean over 10 random halvings of the type's TEST cells (seeds 0-9) of TV(psi_A, psi_B) * sqrt(nA nB) / (nA + nB)
Writes $V/c14_shrinkage/<ds>_reference_units.tsv.gz and <ds>_abs_panel.tsv."""
import os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
from c10_catalog import labels, fit_dir, onehot, tv, MIN_TRAIN  # noqa: E402

ds = sys.argv[1]; OUT = os.path.join(vlib.V, "c14_shrinkage")
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
P = vlib.paths(ds)
z = np.load(os.path.join(fit_dir(ds, 0), "fit.npz"), allow_pickle=True)
inv = np.asarray(z["inv_ev"], np.int64); n_atse = len(z["event_id"]); J = len(inv)
E = sp.csr_matrix((np.ones(J), (np.arange(J), inv)), shape=(J, n_atse))
_, t, _ = labels(ds, P["train"]); Y = vlib.read_counts(P["train"]).tocsr(); N = (Y @ E).tocsr()
_, tt, _ = labels(ds, P["test"]); Yt = vlib.read_counts(P["test"]).tocsr(); Nt = (Yt @ E).tocsr()
cnt = pd.Series(t).value_counts(); types = sorted(cnt[cnt >= MIN_TRAIN].index); code = {x: i for i, x in enumerate(types)}
G = len(types) + 1
gt = np.array([code.get(x, G - 1) for x in tt]); Ot = onehot(gt, G)
yt, nt = np.asarray((Ot @ Yt).todense()), np.asarray((Ot @ Nt).todense())
with np.errstate(invalid="ignore", divide="ignore"):
    tpsi = np.nan_to_num(yt / nt[:, inv])
p = np.asarray(Y.sum(0)).ravel() / np.maximum(np.asarray(N.sum(0)).ravel()[inv], 1e-300)
upper = tv(np.repeat(p[None, :], G, 0), tpsi, inv, n_atse)

fsum = np.zeros((G, n_atse)); fcnt = np.zeros((G, n_atse))
for s in range(10):
    rng = np.random.default_rng(s); half = np.zeros(len(gt), bool)
    for k in range(G - 1):
        i = np.flatnonzero(gt == k); half[rng.permutation(i)[: len(i) // 2]] = True
    pa = []
    for m in (half, ~half):
        O = onehot(gt[m], G); y_, n_ = np.asarray((O @ Yt[m]).todense()), np.asarray((O @ Nt[m]).todense())
        with np.errstate(invalid="ignore", divide="ignore"):
            pa.append((np.nan_to_num(y_ / n_[:, inv]), n_))
    (pA, nA), (pB, nB) = pa
    ok = (nA > 0) & (nB > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = tv(pA, pB, inv, n_atse) * np.sqrt(nA * nB) / (nA + nB)
    fsum[ok] += f[ok]; fcnt[ok] += 1
    log(f"split {s} done")
with np.errstate(invalid="ignore", divide="ignore"):
    floor = np.where(fcnt > 0, fsum / fcnt, np.nan)

U = pd.read_csv(os.path.join(vlib.V, "c10", f"{ds}_s0_accuracy_units.tsv.gz"), sep="\t")
k_ = U.cell_type.map(code).to_numpy(); e_ = U.atse.to_numpy()
U["err_upper"] = upper[k_, e_]; U["err_floor"] = floor[k_, e_]
U[["cell_type", "atse", "size_bin", "read_bin", "err_leafletfa", "err_raw", "err_upper", "err_floor"]].to_csv(
    os.path.join(OUT, f"{ds}_reference_units.tsv.gz"), sep="\t", index=False)
log(f"units {len(U):,}; floor undefined on {int(U.err_floor.isna().sum()):,}")


def boot(sub, col):
    sub = sub[sub[col].notna()]
    rng_ = np.random.default_rng(0); per = sub.groupby("cell_type")[col].agg(["sum", "size"])
    v = []
    for _ in range(1000):
        x = per.iloc[rng_.integers(0, len(per), len(per))]; v.append(x["sum"].sum() / x["size"].sum())
    return sub[col].mean(), *np.quantile(v, [0.025, 0.975])


# share of catalog rows (junction x type, as shipped) per bin; TRAIN cells per type and TRAIN reads at the ATSE
C = pd.read_csv(os.path.join(vlib.V, "c12", "export", f"catalog_{ds}_s0.tsv.gz"), sep="\t", usecols=["cell_type", "cells", "atse_reads"])
C["size_bin"] = pd.cut(C.cells, [9, 49, 199, 999, np.inf], labels=["10-49", "50-199", "200-999", ">=1000"])
C["read_bin"] = pd.cut(C.atse_reads, [-1, 0, 19, 99, np.inf], labels=["0", "1-19", "20-99", ">=100"])
assert C.size_bin.notna().all() and C.read_bin.notna().all()
A10 = pd.read_csv(os.path.join(vlib.V, "tsv", f"c10_{ds}_s0_accuracy.tsv"), sep="\t").set_index(["by", "bin"])
rows = []
for by, col in (("size", "size_bin"), ("reads", "read_bin")):
    share = C[col].value_counts(normalize=True)
    for lev, sub in U.groupby(col, observed=True):
        r = dict(dataset=ds, by=by, bin=str(lev), units=len(sub), types=sub.cell_type.nunique(),
                 catalog_rows_share=float(share.get(lev, 0.0)), floor_units=int(sub.err_floor.notna().sum()))
        for m in ("leafletfa", "raw", "upper", "floor"):
            r[m], r[f"{m}_lo"], r[f"{m}_hi"] = boot(sub, f"err_{m}")
        a = A10.loc[(by, str(lev))]
        assert abs(a.err_leafletfa - r["leafletfa"]) < 1e-9 and abs(a.err_raw - r["raw"]) < 1e-9
        r.update(diff=a["diff"], diff_lo=a.diff_lo, diff_hi=a.diff_hi,
                 pct_reduction=100 * (a.err_raw - a.err_leafletfa) / a.err_raw,
                 diff_ci_excludes_0=bool(a.diff_hi < 0 or a.diff_lo > 0),
                 closed_leafletfa=(r["upper"] - a.err_leafletfa) / (r["upper"] - r["floor"]),   # share of the gap from
                 closed_raw=(r["upper"] - a.err_raw) / (r["upper"] - r["floor"]))                # no cell-type info to the floor
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, f"{ds}_abs_panel.tsv"), sep="\t", index=False)
print(R.round(4).to_string(index=False))
