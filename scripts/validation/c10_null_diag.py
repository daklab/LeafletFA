"""Check 10.2 null, read-only diagnostics (Karin 2026-11-04). The pre-registered verdict stands; nothing here changes it.

    python c10_null_diag.py TMS     # -> c10_null_diag/<ds>_*.tsv(.gz), <ds>_summary.json

Reproduces 10.2a/b on the seed-0 fit: calls from TRUE labels in half A; confirmation in half B under (i) the same rule and
(ii) raw pseudobulk TV > delta / 2 with the same sign at the lead junction. Null (ii): labels permuted among B's cells within
each mouse, the 20 permutations of c10_catalog.py (asserted to reproduce the saved null), confirmed against the SAME half-A
calls. Null (i) comes from the original run's log (20 values); it is not recomputed.
EXPLORATORY additions:
- null (ii) split into P(TV > delta / 2) and P(sign match | TV > delta / 2);
- null (ii) by decile of: per-cell psi variance at the lead junction across B's covered cells (labels ignored), event reads of the
  permuted group, and retained-label fraction of the type (expected share of a permuted group's cells that truly are type t);
- confirmation restricted to units with B reads > 0 and >= 20 (confirm() treats an unread group as psi = 0);
- 5 permutations across all of B's cells, ignoring mouse (mouse-free null);
- type-bootstrap intervals (2,000 resamples of cell types).
"""
import json, os, re, sys, glob
import numpy as np, pandas as pd, scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
import c10_catalog as C  # noqa: E402
from c10_catalog import Data, Groups, type_setup, mouse_halves, eligibility, estimates, reads_cov, raw_pb, call, tv, lead_sign, log, N_PERM  # noqa: E402

OUT = os.path.join(vlib.V, "c10_null_diag")


def rawstat(D, rows, labels_g, G, lj):
    y, n = raw_pb(D.Y, D.Ncell, D.inv, rows, labels_g[rows], G)
    with np.errstate(invalid="ignore", divide="ignore"):
        rp = y[:-1] / n[:-1]; rr = (y.sum(0) - y[:-1]) / (n.sum(0) - n[:-1])
    a, b = np.nan_to_num(rp), np.nan_to_num(rr)
    tvB = tv(a, b, D.inv, D.n_atse)
    sB = np.sign(np.take_along_axis(a - b, np.maximum(lj, 0), axis=1))
    rd, _ = reads_cov(D, rows, labels_g[rows], G)
    return tvB, sB, rd[:-1]


def cell_var(D, rows):
    """per junction: variance of per-cell psi over covered cells of `rows`, labels ignored"""
    Y = D.Y[rows].tocoo(); N = D.Ncell[rows]
    n = np.asarray(N[Y.row, D.inv[Y.col]]).ravel(); p = Y.data / n
    s1 = np.bincount(Y.col, p, D.J); s2 = np.bincount(Y.col, p * p, D.J)
    c = np.asarray((N > 0).sum(0)).ravel()[D.inv].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s2 / c - (s1 / c) ** 2


def boot(tbl, cols, B=2000):
    """type bootstrap of call-weighted means"""
    g = tbl.groupby("cell_type")[cols].sum(); k = tbl.groupby("cell_type").size().values; rng = np.random.default_rng(0); out = {}
    for c in cols:
        v = g[c].values; idx = rng.integers(0, len(v), (B, len(v)))
        r = v[idx].sum(1) / k[idx].sum(1); out[c] = [float(tbl[c].mean()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))]
    return out


def main(ds, seed=0):
    os.makedirs(OUT, exist_ok=True)
    D = Data(ds, seed); types, g = type_setup(D); G = len(types) + 1
    delta = json.load(open(os.path.join(vlib.V, "tsv", f"c10_{ds}_s{seed}_delta.json")))["delta"]
    saved = pd.read_csv(os.path.join(vlib.V, "tsv", f"c10_{ds}_s{seed}_replication.tsv"), sep="\t").iloc[0]
    inA = mouse_halves(D); elig = eligibility(D, g, G - 1, inA); rA, rB = np.flatnonzero(inA), np.flatnonzero(~inA)

    def half(rows, s):
        p, r, *_rest = estimates(D, Groups(D, rows, g[rows], G), draws=True, seed=s, delta=delta)
        P = _rest[2]; rd, cv = reads_cov(D, rows, g[rows], G)
        return dict(p=p[:-1], r=r[:-1], P=P[:-1], reads=rd[:-1], cov=cv[:-1])
    HA = half(rA, seed + 101); HB = half(rB, seed + 202)
    cA, _ = call(D, HA["p"], HA["r"], HA["P"], HA["reads"], HA["cov"], delta, elig)
    lj, sgn = lead_sign(HA["p"], HA["r"], D.inv, D.n_atse)
    cB, _ = call(D, HB["p"], HB["r"], HB["P"], HB["reads"], HB["cov"], delta, elig)
    log(f"calls half A {cA.sum()} (saved {saved.calls_halfA}); confirmed (i) {cB[cA].mean():.4f} (saved {saved.confirmed_i:.4f})")
    K, E = np.nonzero(cA); L = lj[K, E]
    tvB, sB, rdB = rawstat(D, rB, g, G, lj)
    conf = (tvB[K, E] > delta / 2) & (sB[K, E] == sgn[K, E])
    log(f"confirmed (ii) {conf.mean():.4f} (saved {saved.confirmed_ii:.4f})")
    tv_n, sg_n, cf_n, rd_n, zero_n = [np.zeros(len(K)) for _ in range(5)]; per = []; per_pos = []; per20 = []
    for p_ in range(N_PERM):
        rng = np.random.default_rng(1000 + p_); gp = g.copy()
        for m in np.unique(D.mouse[~inA]):
            idx = np.flatnonzero((~inA) & (D.mouse == m)); gp[idx] = rng.permutation(g[idx])
        t_, s_, r_ = rawstat(D, rB, gp, G, lj)
        a = t_[K, E] > delta / 2; b = s_[K, E] == sgn[K, E]; r = r_[K, E]
        tv_n += a; sg_n += a & b; cf_n += a & b; rd_n += r; zero_n += r == 0
        per.append(float((a & b).mean())); per_pos.append(float((a & b)[r > 0].mean())); per20.append(float((a & b)[r >= 20].mean()))
        log(f"  perm {p_}: {per[-1]:.4f}")
    assert abs(np.mean(per) - saved.null_ii) < 2e-3, ("null (ii) not reproduced", np.mean(per), saved.null_ii)
    free = []
    for p_ in range(5):
        rng = np.random.default_rng(2000 + p_); gp = g.copy(); gp[rB] = rng.permutation(g[rB])
        t_, s_, r_ = rawstat(D, rB, gp, G, lj); free.append(float(((t_[K, E] > delta / 2) & (s_[K, E] == sgn[K, E])).mean()))
    # retained-label fraction per type in B
    M = pd.DataFrame(dict(t=g[rB], m=D.mouse[rB])); tab = M.groupby(["t", "m"]).size().unstack(fill_value=0)
    nm = tab.sum(0); ret = {t: float(((tab.loc[t] ** 2) / nm).sum() / tab.loc[t].sum()) for t in tab.index}
    base = {t: float(tab.loc[t].sum() / tab.values.sum()) for t in tab.index}
    var = cell_var(D, rB)
    A = pd.DataFrame(dict(cell_type=np.array(types)[K], type_code=K, atse=E, lead=L, a_reads=HA["reads"][K, E].astype(int), b_reads=rdB[K, E].astype(int),
                          conf_i=cB[K, E].astype(float), conf_ii=conf.astype(float), null_ii=cf_n / N_PERM, null_tv=tv_n / N_PERM,
                          null_perm_reads=rd_n / N_PERM, null_zero_read=zero_n / N_PERM, cell_var=var[L],
                          retained=[ret.get(k, np.nan) for k in K], base_rate=[base.get(k, np.nan) for k in K]))
    A.to_csv(os.path.join(OUT, f"{ds}_calls.tsv.gz"), sep="\t", index=False, float_format="%.5g")
    rows = []
    for col in ("cell_var", "null_perm_reads", "retained"):
        q = pd.qcut(A[col].rank(method="first"), 10, labels=False) + 1
        for d, X in A.groupby(q):
            rows.append(dict(by=col, decile=int(d), calls=len(X), lo=float(X[col].min()), hi=float(X[col].max()), real_ii=X.conf_ii.mean(), real_i=X.conf_i.mean(),
                             null_ii=X.null_ii.mean(), null_tv=X.null_tv.mean(), sign_given_tv=X.null_ii.sum() / max(X.null_tv.sum(), 1e-12),
                             share_of_null_confirmations=X.null_ii.sum() / A.null_ii.sum()))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{ds}_deciles.tsv"), sep="\t", index=False, float_format="%.5g")
    logf = sorted(glob.glob(os.path.join(vlib.V, "logs", f"lfa_c10_{ds}_s0_[0-9]*.log")))[-1]
    nulli = [float(x) for x in re.findall(r"null \d+: \(([0-9.e-]+),", open(logf).read())][-N_PERM:]
    mice_A = pd.DataFrame(dict(t=g[rA], m=D.mouse[rA])).groupby("t").m.nunique(); mice_B = pd.DataFrame(dict(t=g[rB], m=D.mouse[rB])).groupby("t").m.nunique()
    el = np.flatnonzero(elig)
    S = dict(dataset=ds, delta=delta, mice_total=int(len(np.unique(D.mouse))), calls_halfA=int(cA.sum()), saved_calls_halfA=int(saved.calls_halfA), bootstrap=boot(A, ["conf_i", "conf_ii", "null_ii"]),
             saved=dict(confirmed_i=float(saved.confirmed_i), confirmed_ii=float(saved.confirmed_ii), null_i=float(saved.null_i), null_ii=float(saved.null_ii)),
             null_ii_perm_range=[min(per), max(per)], null_i_perm_range=[min(nulli), max(nulli)], null_i_source=os.path.basename(logf), n_null_i=len(nulli),
             null_tv_rate=float(A.null_tv.mean()), sign_match_given_tv=float(A.null_ii.sum() / A.null_tv.sum()),
             null_ii_reads_gt0=float(np.mean(per_pos)), null_ii_reads_ge20=float(np.mean(per20)), null_zero_read_share=float(A.null_zero_read.mean()),
             real_ii_b_reads_gt0=float(A.conf_ii[A.b_reads > 0].mean()), real_ii_b_reads_ge20=float(A.conf_ii[A.b_reads >= 20].mean()), real_zero_read_share=float((A.b_reads == 0).mean()),
             null_ii_mouse_free_5perm=free, retained_median=float(A.retained.median()), base_rate_median=float(A.base_rate.median()),
             leafcutter_types_ge4mice_each_half=int(sum(mice_A.get(k, 0) >= 4 and mice_B.get(k, 0) >= 4 for k in el)),
             eligible_types=int(len(el)), median_mice_half_A=float(np.median([mice_A.get(k, 0) for k in el])))
    json.dump(S, open(os.path.join(OUT, f"{ds}_summary.json"), "w"), indent=1); print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
