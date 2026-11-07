"""Check 13 (amendment 8, Karin 2026-11-06): group-pooled cell-type estimate vs current LeafletFA and covered-cells-only.
    python c13_pooled.py <TMS|Allen>
Read-only from the seed-0 fit, TRAIN and TEST counts. Estimates (G-1 types x J junctions):
  current  = saved check 10 point (mean of rho~ over the type's TRAIN cells; asserted against a recompute)
  covered  = mean of rho~ over the type's TRAIN cells with N_ce > 0; program prior rhobar where none is covered
  pooled   = (g_e rhobar + sum y) / (g_e + sum N), rhobar = mean over the type's cells of phi_c psi; Beta interval
Readouts (1) browser-gap table as c10_browser_gap_rank.py, (2) 10.1 accuracy, (3) pooled Beta coverage on the 10.3 units.
Writes $V/c13_pooled/<ds>_*.tsv and <ds>_summary.json."""
import os, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import beta as Beta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
from c10_catalog import (Data, Groups, type_setup, estimates, raw_pb, onehot, tv, log,  # noqa: E402
                         MIN_TEST_CELLS, MIN_TEST_READS, MIN_TEST_COV)
ds = sys.argv[1]; OUT = os.path.join(vlib.V, "c13_pooled"); os.makedirs(OUT, exist_ok=True)
D = Data(ds, 0); D.load_test(ds); types, g = type_setup(D); G = len(types) + 1; allrows = np.arange(len(g))
E = np.load(os.path.join(vlib.V, "c10", f"{ds}_s0_estimates.npz"), allow_pickle=True); assert list(E["types"].astype(str)) == list(types)
Gr = Groups(D, allrows, g, G)
p, *_ = estimates(D, Gr, draws=False); assert float(np.abs(p[:-1] - E["point"]).max()) < 1e-4
cur = E["point"].astype(np.float64); n = Gr.n[:-1]
phi, psi = np.asarray(D.post["phi"], np.float64), np.asarray(D.post["psi_mean"], np.float64)   # cells x K, K x J
Phis = Gr.O @ phi                                                                            # G x K
rhobar = (Phis[:-1] / n[:, None]) @ psi
y, nn = raw_pb(D.Y, D.Ncell, D.inv, allrows, g, G); y, nn = y[:-1], nn[:-1]                  # junction-level sums
gam = D.gam[D.inv][None, :]
a_par = gam * rhobar + y; b_par = gam * (1 - rhobar) + nn - y
pooled = a_par / (gam + nn)
cov_cells = E["cov"]                                                                          # types x ATSE, covered TRAIN cells
covd = np.empty_like(cur)
for k in range(G - 1):
    i = Gr.idx[k]; Ck = ((D.Ncell[i] > 0).astype(np.float64).T @ phi[i])                     # ATSE x K, sum of phi over covered cells
    unc = np.einsum("jk,kj->j", Phis[k][None, :] - Ck[D.inv], psi)                           # sum over uncovered cells of prior
    nc = cov_cells[k][D.inv]
    with np.errstate(invalid="ignore", divide="ignore"):
        covd[k] = np.where(nc > 0, (cur[k] * n[k] - unc) / nc, rhobar[k])
# direct per-cell check of covered and rhobar on the smallest type
k0 = int(np.argmin(n)); i = Gr.idx[k0]
Y0 = D.Y[i].toarray().astype(np.float64); N0 = D.Ncell[i].toarray()[:, D.inv].astype(np.float64)
V0 = D.V[i].toarray()[:, D.inv]; R0 = D.R[i].toarray(); pr0 = phi[i] @ psi; rho0 = (1 - V0) * pr0 + R0; c0 = N0 > 0
with np.errstate(invalid="ignore", divide="ignore"):
    direct = np.where(c0.sum(0) > 0, np.where(c0, rho0, 0).sum(0) / c0.sum(0), pr0.mean(0))
assert np.allclose(direct, covd[k0], atol=1e-5) and np.allclose(pr0.mean(0), rhobar[k0], atol=1e-6) and np.allclose(rho0.mean(0), cur[k0], atol=1e-4)
s_atse = lambda X: tv(X, np.zeros_like(X), D.inv, D.n_atse) * 2                              # per-ATSE sums of a nonneg table
assert np.allclose(s_atse(pooled[:3]), 1, atol=1e-6) and np.allclose(s_atse(covd[:3]), 1, atol=1e-4)
log(f"{ds}: asserts passed (current = saved point; covered and rhobar match a per-cell rebuild on {types[k0]}; ATSE sums = 1)")
with np.errstate(invalid="ignore", divide="ignore"):
    TR = y / nn
EST = {"current": cur, "covered-only": covd, "pooled": pooled, "raw TRAIN": TR}
# TEST
te = D.test; code = {t: i for i, t in enumerate(types)}; gt = np.array([code.get(t, G - 1) for t in te["t"]])
Ot = onehot(gt, G); nte_cells = np.bincount(gt, minlength=G)
yt = np.asarray((Ot @ te["Y"]).todense())[:-1]; nt_a = np.asarray((Ot @ te["N"]).todense())[:-1]
ct = np.asarray((Ot @ (te["N"] > 0).astype(np.float64)).todense())[:-1]; nt_j = nt_a[:, D.inv]
with np.errstate(invalid="ignore", divide="ignore"):
    TE = yt / nt_j
# ---- (1) browser-gap table ----
TRe, TRc = E["reads"], E["cov"]
rng = np.random.default_rng(7); elig = np.flatnonzero(E["eligible"])
ex = {"TMS": ("Brain_Non-Myeloid_NEURON", "Brain_Non-Myeloid_ASTROCYTE"), "Allen": ("L2/3 IT CTX", "Pvalb")}[ds]
pairs = [(code[ex[0]], code[ex[1]])]
while len(pairs) < 21:
    a, b = rng.choice(elig, 2, replace=False)
    if (a, b) not in pairs: pairs.append((int(a), int(b)))
rows = []
for a, b in pairs:
    ok = (TRe[a] >= 20) & (TRc[a] >= 3) & (TRe[b] >= 20) & (TRc[b] >= 3)
    dTR = TR[a] - TR[b]; dTE = TE[a] - TE[b]
    for name, X in EST.items():
        dX = X[a] - X[b]
        f = pd.DataFrame(dict(e=D.inv, k=np.abs(dX), j=np.arange(D.J)))[ok[D.inv]].sort_values("k", ascending=False).drop_duplicates("e")
        f["rank"] = np.arange(1, len(f) + 1); j = f.j.values; e = f.e.values
        f["d"], f["dTR"], f["dTE"] = dX[j], dTR[j], dTE[j]
        f["te_ok"] = (nt_a[a, e] >= 20) & (ct[a, e] >= 3) & (nt_a[b, e] >= 20) & (ct[b, e] >= 3)
        for lab, m in (("top100", f["rank"] <= 100), ("101-1000", (f["rank"] > 100) & (f["rank"] <= 1000)), ("all", f["rank"] > 0)):
            x = f[m]; z = x[x.te_ok]
            rows.append(dict(dataset=ds, pair=f"{types[a]} vs {types[b]}", example=(a, b) == pairs[0], estimate=name, bin=lab, n=len(x), n_test=len(z),
                             gap_vs_TR=float((x.d - x.dTR).abs().median()), err_vs_TE=float((z.d - z.dTE).abs().median()),
                             absd_minus_absTE=float((z.d.abs() - z.dTE.abs()).median()), sign_TE=float((np.sign(z.d) == np.sign(z.dTE)).mean())))
R1 = pd.DataFrame(rows); R1.to_csv(os.path.join(OUT, f"{ds}_rank_table.tsv"), sep="\t", index=False, float_format="%.4f")
num = ["n_test", "gap_vs_TR", "err_vs_TE", "absd_minus_absTE", "sign_TE"]
R1m = R1[~R1.example].groupby(["bin", "estimate"], sort=False)[num].median()
R1m.to_csv(os.path.join(OUT, f"{ds}_rank_median20.tsv"), sep="\t", float_format="%.4f")
log("browser-gap, 20 random pairs, median over pairs:\n" + R1m.round(3).to_string())
log("example pair:\n" + R1[R1.example].set_index(["bin", "estimate"])[num].round(3).to_string())
# ---- (2) 10.1 accuracy ----
yr, nr = raw_pb(D.Y, D.Ncell, D.inv, allrows, g, G)
pop = yr.sum(0) / np.maximum(nr.sum(0), 1e-12)
with np.errstate(invalid="ignore", divide="ignore"):
    raw = np.where(nr > 0, yr / np.maximum(nr, 1e-12), pop[None, :])[:-1]
tp0 = np.nan_to_num(TE)
err = {k: tv(X, tp0, D.inv, D.n_atse) for k, X in (("current", cur), ("covered-only", covd), ("pooled", pooled), ("raw", raw))}
U = []
for k, t in enumerate(types):
    if nte_cells[k] < MIN_TEST_CELLS:
        continue
    u = np.flatnonzero((nt_a[k] >= MIN_TEST_READS) & (ct[k] >= MIN_TEST_COV))
    U.append(pd.DataFrame(dict(cell_type=t, atse=u, train_cells=int(n[k]), train_reads=TRe[k, u], **{f"err_{m}": err[m][k, u] for m in err})))
U = pd.concat(U, ignore_index=True)
U["size_bin"] = pd.cut(U.train_cells, [9, 49, 199, 999, np.inf], labels=["10-49", "50-199", "200-999", ">=1000"])
U["read_bin"] = pd.cut(U.train_reads, [-1, 0, 19, 99, np.inf], labels=["0", "1-19", "20-99", ">=100"])
S0 = pd.read_csv(os.path.join(vlib.V, "c10", f"{ds}_s0_accuracy_units.tsv.gz"), sep="\t")
assert len(S0) == len(U) and abs(S0.err_leafletfa.mean() - U.err_current.mean()) < 1e-5 and abs(S0.err_raw.mean() - U.err_raw.mean()) < 1e-5, "10.1 units not reproduced"


def boot(sub, col):
    r = np.random.default_rng(0); per = sub.groupby("cell_type")[col].agg(["sum", "size"])
    v = [x["sum"].sum() / x["size"].sum() for x in (per.iloc[r.integers(0, len(per), len(per))] for _ in range(1000))]
    return np.quantile(v, [0.025, 0.975])


for m in ("current", "covered-only", "pooled"):
    U[f"d_{m}_raw"] = U[f"err_{m}"] - U.err_raw
U["d_pooled_current"] = U.err_pooled - U.err_current; U["d_covered_current"] = U["err_covered-only"] - U.err_current
acc = []
for kind, col in (("all", None), ("size", "size_bin"), ("reads", "read_bin")):
    for lev, sub in ([("all", U)] if col is None else U.groupby(col, observed=True)):
        r = dict(dataset=ds, by=kind, bin=str(lev), units=len(sub), types=sub.cell_type.nunique(),
                 **{f"err_{m}": sub[f"err_{m}"].mean() for m in ("current", "covered-only", "pooled", "raw")})
        for dcol in ("d_current_raw", "d_covered-only_raw", "d_pooled_raw", "d_pooled_current", "d_covered_current"):
            lo, hi = boot(sub, dcol); r.update({dcol: sub[dcol].mean(), dcol + "_lo": lo, dcol + "_hi": hi})
        acc.append(r)
A = pd.DataFrame(acc); A.to_csv(os.path.join(OUT, f"{ds}_accuracy.tsv"), sep="\t", index=False, float_format="%.5f")
log("10.1-style accuracy:\n" + A[["by", "bin", "units", "err_current", "err_covered-only", "err_pooled", "err_raw",
                                  "d_pooled_current", "d_pooled_current_lo", "d_pooled_current_hi"]].round(4).to_string(index=False))
# ---- (3) pooled Beta coverage on the 10.3 units ----
sav = pd.read_csv(os.path.join(vlib.V, "tsv", f"c10_{ds}_s0_calibration.tsv"), sep="\t", index_col=0)
cal = []
for k, t in enumerate(types):
    for lo_b, hi_b in ((100, 499), (500, np.inf)):
        u = np.flatnonzero((nt_a[k] >= lo_b) & (nt_a[k] <= hi_b) & (ct[k] >= 10))
        j = np.flatnonzero(np.isin(D.inv, u) & (nt_j[k] > 0))
        if not len(j):
            continue
        lo, hi = Beta.ppf(0.025, a_par[k, j], b_par[k, j]), Beta.ppf(0.975, a_par[k, j], b_par[k, j])
        trr = TRe[k, D.inv[j]]
        cal.append(pd.DataFrame(dict(cell_type=t, test_read_bin=f"{lo_b}-{hi_b}", j=j, inside=(tp0[k, j] >= lo) & (tp0[k, j] <= hi), width=hi - lo,
                                     train_read_bin=pd.cut(trr, [-1, 0, 99, 999, np.inf], labels=["0", "1-99", "100-999", ">=1000"]).astype(str))))
C = pd.concat(cal, ignore_index=True); C.to_csv(os.path.join(OUT, f"{ds}_coverage_units.tsv.gz"), sep="\t", index=False)
Cs = C.groupby("test_read_bin").agg(junctions=("inside", "size"), coverage=("inside", "mean"), median_width=("width", "median"))
assert (Cs.junctions.values == sav.loc[Cs.index, "junctions"].values).all(), "10.3 units not reproduced"
Cs["check10_coverage_bootstrap"] = sav.loc[Cs.index, "coverage_bootstrap"].values
Ct = C.groupby(["test_read_bin", "train_read_bin"]).agg(junctions=("inside", "size"), coverage=("inside", "mean"), median_width=("width", "median"))
Cs.to_csv(os.path.join(OUT, f"{ds}_coverage.tsv"), sep="\t", float_format="%.4f"); Ct.to_csv(os.path.join(OUT, f"{ds}_coverage_by_train_reads.tsv"), sep="\t", float_format="%.4f")
log("pooled Beta coverage:\n" + Cs.round(4).to_string() + "\n" + Ct.round(4).to_string())
a_all = A[A.by == "all"].iloc[0]
S = dict(dataset=ds, top100_err_vs_TE={m: float(R1m.loc[("top100", m), "err_vs_TE"]) for m in EST},
         reading_a_pooled_lower_top100=bool(R1m.loc[("top100", "pooled"), "err_vs_TE"] < R1m.loc[("top100", "current"), "err_vs_TE"]),
         overall=dict(err_current=a_all.err_current, err_pooled=a_all.err_pooled, err_covered=a_all["err_covered-only"], err_raw=a_all.err_raw,
                      d_pooled_current=[a_all.d_pooled_current, a_all.d_pooled_current_lo, a_all.d_pooled_current_hi]),
         reading_b_pooled_no_worse=bool(a_all.d_pooled_current_lo <= 0),
         coverage={b: float(Cs.loc[b, "coverage"]) for b in Cs.index},
         coverage_within_0_05={b: bool(0.90 <= Cs.loc[b, "coverage"] <= 1.0) for b in Cs.index})
json.dump(S, open(os.path.join(OUT, f"{ds}_summary.json"), "w"), indent=1, default=float); log(json.dumps(S, indent=1, default=float))
