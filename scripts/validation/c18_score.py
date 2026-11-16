"""Check 18 (amendment 14, CPU): calls from the half-A refit, confirmation only in half-B mice (TMS).
    python c18_score.py lc_prepare   # LeafCutter inputs for check 17's 50-pair subset from half-A cells -> $V/c18/lc/
    python c18_score.py score        # 18a pairwise (check 17 design) and 18b type vs rest (cap 5 per type), rule, rank bands
"""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, c9_lib as L  # noqa: E402
from c10_catalog import Groups, onehot, lead_sign, log  # noqa: E402
from c9_ds_null import lc_name  # noqa: E402

V18 = os.path.join(vlib.V, "c18"); LCD = os.path.join(V18, "lc"); FIT = os.path.join(V18, "fit_halfA", "fit")
DELTA = 0.10; CAP = 5; N_PERM = 20; MIN_UNITS = 10
BINS = ["1-19", "20-99", "100-499", ">=500"]; COMPARED = ["20-99", "100-499", ">=500"]
LF, RAW, LC = "LeafletFA", "raw_pseudobulk", "LeafCutter"; BANDS = [(1, 5, "1-5"), (6, 20, "6-20"), (21, 100, "21-100")]


def slug(t):
    return "".join(c if c.isalnum() else "_" for c in t)


def binof(r):
    return np.select([r <= 19, r <= 99, r <= 499], ["1-19", "20-99", "100-499"], ">=500")


class Half:
    """counts and labels of one half; with post, the per-cell quantities of c10_catalog.Data"""
    def __init__(self, h, var, post=None):
        C = pd.read_parquet(os.path.join(V18, "cells.parquet")).set_index("cell")
        self.cells = pd.read_csv(os.path.join(V18, f"half{h}_cells.tsv"), sep="\t").cell.values.astype(str)
        self.t = C.loc[self.cells, "cell_type"].values.astype(str); self.mouse = C.loc[self.cells, "mouse"].values.astype(str)
        self.Y = sp.load_npz(os.path.join(V18, f"half{h}_counts.npz")).tocsr().astype(np.float64)
        jid, ev = var; _, self.inv = np.unique(ev, return_inverse=True); self.n_atse = self.inv.max() + 1; self.J = len(self.inv)
        if post is not None:
            assert np.array_equal(post["cell_id"], self.cells) and np.array_equal(post["inv"].numpy(), self.inv)
            self.post = post
        E = sp.csr_matrix((np.ones(self.J), (np.arange(self.J), self.inv)), shape=(self.J, self.n_atse))
        self.Ncell = (self.Y @ E).tocsr()
        if post is not None:
            self.gam = np.asarray(np.load(os.path.join(FIT, "fit.npz"), allow_pickle=True)["bb_conc_vec"], np.float64)
            Nc = self.Ncell.tocoo(); Yc = self.Y.tocoo(); Ncj = np.asarray(self.Ncell[Yc.row, self.inv[Yc.col]]).ravel()
            self.V = sp.csr_matrix((Nc.data / (self.gam[Nc.col] + Nc.data), (Nc.row, Nc.col)), shape=self.Ncell.shape)
            self.R = sp.csr_matrix((Yc.data / (self.gam[self.inv[Yc.col]] + Ncj), (Yc.row, Yc.col)), shape=self.Y.shape)

    def pb(self, lab, G):
        O = onehot(lab, G); return np.asarray((O @ self.Y).todense()), np.asarray((O @ self.Ncell).todense())


def setup(post=False):
    var = vlib.read_var(os.path.join(V18, "var.tsv"))
    P = pd.read_csv(os.path.join(vlib.V, "c17", "pairs.tsv"), sep="\t")
    keep = pd.read_csv(os.path.join(vlib.V, "c10_lc", "TMS_halfA", "types.tsv"), sep="\t").type.values
    if post:
        old = L.fit_dir; L.fit_dir = lambda s: FIT
        try:
            po = L.load_posterior(0)
        finally:
            L.fit_dir = old
        return var, P, keep, Half("A", var, po), Half("B", var)
    return var, P, keep, Half("A", var), None


def lc_prepare():
    var, P, keep, A, _ = setup(); os.makedirs(LCD, exist_ok=True)
    names = np.array([lc_name(j, e) for j, e in zip(*var)]); mice = np.unique(A.mouse); mi = {m: i for i, m in enumerate(mice)}
    mc = np.array([mi[m] for m in A.mouse]); jobs = []; rec = []
    for x in P[P.lc_subset].itertuples():
        cols, grp, M = [], [], []
        for t, arm in ((x.A, "armT"), (x.B, "armR")):
            s = np.flatnonzero(A.t == t)
            Mo = sp.csr_matrix((np.ones(len(s)), (mc[s], np.arange(len(s)))), shape=(len(mice), len(s))); T = np.asarray((Mo @ A.Y[s]).todense())
            for i in np.flatnonzero(np.bincount(mc[s], minlength=len(mice)) > 0):
                cols.append(f"{mice[i]}_{arm}"); grp.append(arm); M.append(T[i])
        rec.append(dict(A=x.A, B=x.B, samples_A=grp.count("armT"), samples_B=grp.count("armR")))
        dd = os.path.join(LCD, f"{slug(x.A)}__{slug(x.B)}"); os.makedirs(dd, exist_ok=True)
        if M:
            M = np.vstack(M).T; ok = M.sum(1) > 0
            pd.DataFrame(M[ok].astype(int), index=names[ok], columns=cols).to_csv(os.path.join(dd, "counts.tsv.gz"), sep="\t", compression="gzip", index_label="chrom")
            open(os.path.join(dd, "groups.txt"), "w").write("".join(f"{c}\t{a}\n" for c, a in zip(cols, grp))); jobs.append(dd)
    open(os.path.join(LCD, "array_jobs.txt"), "w").write("\n".join(jobs) + "\n"); pd.DataFrame(rec).to_csv(os.path.join(LCD, "pairs_samples.tsv"), sep="\t", index=False)
    log(f"LeafCutter inputs: {len(jobs)} pairs; pairs with < 4 half-A mice in a type: {int(((pd.DataFrame(rec)[['samples_A', 'samples_B']] < 4).any(axis=1)).sum())}")


def tvj(d, inv, n_atse):
    out = np.zeros(d.shape[:-1] + (n_atse,)); np.add.at(out.T, inv, np.abs(d).T); return 0.5 * out


def ordered(t):
    return t.iloc[np.lexsort((t.atse.values, -t.tie.values, -t.stat.values))]


def boot(cm, units):
    units = np.array(sorted(units)); pos = {k: i for i, k in enumerate(units)}
    br = np.random.default_rng(0).integers(0, len(units), (2000, len(units))); w = np.stack([np.bincount(r, minlength=len(units)) for r in br])
    res = {}
    for m, t in cm.items():
        i = t.grp.map(pos).values; s_ = np.bincount(i, t.excess.values, len(units)); c_ = np.bincount(i, minlength=len(units)).astype(float)
        res[m] = (w @ s_) / np.maximum(w @ c_, 1e-12)
    return res


def capped(A, comps, scopes):
    summ, diffs = [], []
    for scope, cls in scopes:
        for b in BINS:
            for comp, lc_only in comps:
                sub = A[(A.bin == b) & A.scorable & A.method.isin([LF, comp])]
                if cls:
                    sub = sub[sub.pair_class == cls]
                if lc_only:
                    sub = sub[sub.lc_subset]
                n = sub.groupby(["grp", "method"]).size().unstack(1).reindex(columns=[LF, comp]).fillna(0)
                us = n.index[(n >= CAP).all(1)].values
                cm = {m: ordered(sub[(sub.method == m) & sub.grp.isin(us)]).groupby("grp", sort=False).head(CAP) for m in (LF, comp)}
                row = dict(scope=scope, bin=b, comparator=comp, units=len(us), compared=len(us) >= MIN_UNITS)
                if len(us):
                    bx = boot(cm, us); dd = bx[LF] - bx[comp]
                    for m in (LF, comp):
                        summ.append(dict(scope=scope, bin=b, comparison=f"{LF} vs {comp}", method=m, units=len(us), calls=len(cm[m]), real=cm[m].real.mean(),
                                         null=cm[m].null.mean(), excess=cm[m].excess.mean(), excess_lo=np.quantile(bx[m], .025), excess_hi=np.quantile(bx[m], .975)))
                    row.update(diff=cm[LF].excess.mean() - cm[comp].excess.mean(), diff_lo=np.quantile(dd, .025), diff_hi=np.quantile(dd, .975), lf_excess_lo=np.quantile(bx[LF], .025))
                diffs.append(row)
    return pd.DataFrame(summ), pd.DataFrame(diffs)


def rule(Df, comps):
    Pm = Df[(Df.scope == "all") & Df.bin.isin(COMPARED) & Df.compared]
    noninf = bool(len(Pm) and (Pm.diff_lo > -0.05).all())
    better = noninf and all((Pm[Pm.comparator == c].diff_lo > 0).any() for c in comps if (Pm.comparator == c).any())
    th = Df[(Df.scope == "all") & (Df.bin == "1-19") & (Df.comparator == RAW)]
    thin = bool(len(th) and th.compared.iloc[0] and th.lf_excess_lo.iloc[0] > 0 and th.diff_lo.iloc[0] > 0)
    return dict(compared=[f"{r.bin}:{r.comparator} ({r.units})" for r in Pm.itertuples()],
                not_compared=[f"{r.bin}:{r.comparator} ({r.units})" for r in Df[(Df.scope == "all") & Df.bin.isin(COMPARED) & ~Df.compared].itertuples()],
                non_inferior=noninf, better=better, verdict="BETTER" if better else ("NON-INFERIOR" if noninf else "FAIL"), thin_1_19_supported=thin)


def rank_bands(A):
    rb = []
    for m in (LF, RAW):
        t = A[(A.method == m) & A.scorable].copy(); t = t.iloc[np.lexsort((t.atse.values, -t.tie.values, -t.stat.values, t.grp.values))]
        t["rank"] = t.groupby("grp").cumcount() + 1
        for lo, hi, bl in BANDS:
            for b, u in t[(t["rank"] >= lo) & (t["rank"] <= hi)].groupby("bin"):
                rb.append(dict(method=m, rank_band=bl, bin=b, calls=len(u), units=u.grp.nunique(), confirmed=u.real.mean(), null=u.null.mean(), excess=u.excess.mean()))
    return pd.DataFrame(rb)


def score():
    t0 = time.time(); var, P, keep, A, B = setup(post=True)
    nA = pd.Series(A.t).value_counts(); types = sorted(nA[nA >= 10].index); assert set(keep) <= set(types), "a check 16 type has < 10 half-A cells"
    code = {t: i for i, t in enumerate(types)}; G = len(types) + 1
    gA = np.array([code.get(t, len(types)) for t in A.t]); gB = np.array([code.get(t, len(types)) for t in B.t])
    Gr = Groups(A, np.arange(len(gA)), gA, G); S, _ = Gr.sums(A, A.post["phi"], A.post["psi_mean"])
    point = S / Gr.n[:, None]; rest = (S.sum(0)[None, :] - S) / (Gr.n.sum() - Gr.n)[:, None]
    yA, nA_ = A.pb(gA, G)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.nan_to_num(yA / nA_[:, A.inv]); rraw = np.nan_to_num((yA.sum(0) - yA) / (nA_.sum(0) - nA_)[:, A.inv])
    readsA = nA_; readsB = np.asarray((onehot(gB, G) @ B.Ncell).todense()); covB = np.asarray((onehot(gB, G) @ (B.Ncell > 0).astype(np.float64)).todense())
    jname = {lc_name(j, e): i for i, (j, e) in enumerate(zip(*var))}; out = []
    BY, BN = B.Y, B.Ncell

    def pbB(rows_, lab, k):
        r = rows_[lab == k]; yy = np.asarray(BY[r].sum(0)).ravel(); nb = np.asarray(BN[r].sum(0)).ravel()[B.inv]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nan_to_num(yy / nb)

    def conf(d, E, lj, sg):
        tvv = tvj(d, B.inv, B.n_atse); return (tvv[E] > DELTA / 2) & (np.sign(d[np.maximum(lj, 0)]) == sg)
    # ---- 18a pairwise ----
    for pi, x in enumerate(P.itertuples()):
        a, b = code.get(x.A), code.get(x.B)
        if a is None or b is None:
            continue
        E = np.flatnonzero((readsA[a] >= 1) & (readsA[b] >= 1) & (readsB[a] >= 20) & (readsB[b] >= 20) & (covB[a] >= 3) & (covB[b] >= 3))
        if not len(E):
            continue
        bn = binof(np.minimum(readsA[a, E], readsA[b, E])); meth = {}
        for nm, X in ((LF, point), (RAW, raw)):
            lj, sg = lead_sign(X[[a]], X[[b]], A.inv, A.n_atse); lj, sg = lj[0, E], sg[0, E]
            meth[nm] = (np.abs(X[a, np.maximum(lj, 0)] - X[b, np.maximum(lj, 0)]), np.zeros(len(E)), lj, sg, np.ones(len(E), bool))
        if x.lc_subset:
            dd = os.path.join(LCD, f"{slug(x.A)}__{slug(x.B)}"); fs, fe = os.path.join(dd, "ds_cluster_significance.txt"), os.path.join(dd, "ds_effect_sizes.txt")
            st = np.full(len(E), np.nan); tie = np.zeros(len(E)); lj = np.full(len(E), -1); sg = np.zeros(len(E))
            if os.path.exists(fs) and os.path.exists(fe):
                c = pd.read_csv(fs, sep="\t"); c = c[c.status == "Success"]; e = pd.read_csv(fe, sep="\t")
                if len(e):
                    e["j"] = e.intron.map(jname); assert e.j.notna().all()
                    e["cl"] = e.intron.str.split(":").str[0] + ":" + e.intron.str.split(":").str[3]; e = e[e.cl.isin(set(c.cluster))]; e["a"] = A.inv[e.j.astype(int)]
                    pc = dict(zip(c.cluster, c.p)); pos = {u: i for i, u in enumerate(E)}
                    for at, yy in e.groupby("a"):
                        if at in pos:
                            l = yy.loc[yy.deltapsi_armT.abs().idxmax()]; i = pos[at]
                            st[i], tie[i], lj[i], sg[i] = -pc[l.cl], abs(l.deltapsi_armT), int(l.j), np.sign(l.deltapsi_armT)
            meth[LC] = (st, tie, lj, sg, ~np.isnan(st))
        rows_ = np.flatnonzero(np.isin(gB, [a, b])); lab = gB[rows_]
        d0 = pbB(rows_, lab, a) - pbB(rows_, lab, b); nulls = []
        for p in range(N_PERM):
            rg = np.random.default_rng(1000 + p); lp = lab.copy()
            for m in np.unique(B.mouse[rows_]):
                i = np.flatnonzero(B.mouse[rows_] == m); lp[i] = rg.permutation(lab[i])
            nulls.append(pbB(rows_, lp, a) - pbB(rows_, lp, b))
        for nm, (st, tie, lj, sg, ok) in meth.items():
            real = conf(d0, E, lj, sg); nl = np.mean([conf(u, E, lj, sg) for u in nulls], axis=0)
            out.append(pd.DataFrame(dict(design="18a", grp=pi, pair_class=x.pair_class, lc_subset=x.lc_subset, atse=E, bin=bn, method=nm, stat=st, tie=tie,
                                         scorable=ok, real=real.astype(float), null=nl)))
        if pi % 50 == 0:
            log(f"pair {pi}/{len(P)}")
    # ---- 18b type vs rest ----
    yB, nB = B.pb(gB, G)

    def dB(lab):
        y_, n_ = B.pb(lab, G)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nan_to_num(y_[:-1] / n_[:-1, B.inv]) - np.nan_to_num((y_.sum(0) - y_[:-1]) / (n_.sum(0) - n_[:-1])[:, B.inv])
    dR = dB(gB); nullsB = []
    for p in range(N_PERM):
        rg = np.random.default_rng(1000 + p); gp = gB.copy()
        for m in np.unique(B.mouse):
            i = np.flatnonzero(B.mouse == m); gp[i] = rg.permutation(gB[i])
        nullsB.append(dB(gp)); log(f"18b null {p}")
    tvR = tvj(dR, B.inv, B.n_atse); tvN = [tvj(u, B.inv, B.n_atse) for u in nullsB]
    for t in keep:
        k = code[t]; E = np.flatnonzero((readsA[k] >= 1) & (readsB[k] >= 20) & (covB[k] >= 3))
        if not len(E):
            continue
        bn = binof(readsA[k, E])
        for nm, X, Xr in ((LF, point, rest), (RAW, raw, rraw)):
            lj, sg = lead_sign(X[[k]], Xr[[k]], A.inv, A.n_atse); lj, sg = lj[0, E], sg[0, E]
            st = np.abs(X[k, np.maximum(lj, 0)] - Xr[k, np.maximum(lj, 0)])
            real = (tvR[k, E] > DELTA / 2) & (np.sign(dR[k, np.maximum(lj, 0)]) == sg)
            nl = np.mean([(tn[k, E] > DELTA / 2) & (np.sign(dn[k, np.maximum(lj, 0)]) == sg) for tn, dn in zip(tvN, nullsB)], axis=0)
            out.append(pd.DataFrame(dict(design="18b", grp=k, pair_class="type_vs_rest", lc_subset=False, atse=E, bin=bn, method=nm, stat=st, tie=0.0,
                                         scorable=True, real=real.astype(float), null=nl)))
    U = pd.concat(out, ignore_index=True); U["excess"] = U.real - U.null; U.to_parquet(os.path.join(V18, "units.parquet"), index=False)
    Ua, Ub = U[U.design == "18a"], U[U.design == "18b"]
    Sa, Da = capped(Ua, [(RAW, False), (LC, True)], [("all", None)] + [(k, k) for k in ("same_tissue", "same_cell_type", "unrelated")])
    Sb, Db = capped(Ub, [(RAW, False)], [("all", None)])
    for nm, S, D in (("18a", Sa, Da), ("18b", Sb, Db)):
        S.to_csv(os.path.join(V18, f"{nm}_summary.tsv"), sep="\t", index=False, float_format="%.4f"); D.to_csv(os.path.join(V18, f"{nm}_differences.tsv"), sep="\t", index=False, float_format="%.4f")
        rank_bands(U[U.design == nm]).to_csv(os.path.join(V18, f"{nm}_rank_bands.tsv"), sep="\t", index=False, float_format="%.4f")
    lc_out = sum(os.path.exists(os.path.join(LCD, f"{slug(x.A)}__{slug(x.B)}", "ds_cluster_significance.txt")) for x in P[P.lc_subset].itertuples())
    Vd = dict(design_18a=rule(Da, (RAW, LC)), design_18b=rule(Db, (RAW,)), pairs_scored=int(Ua.grp.nunique()), types_scored=int(Ub.grp.nunique()),
              lc_outputs=int(lc_out), half_A_cells=len(A.t), half_B_cells=len(B.t), minutes=(time.time() - t0) / 60)
    json.dump(Vd, open(os.path.join(V18, "verdict.json"), "w"), indent=1)
    log(Da[Da.scope == "all"].round(3).to_string(index=False)); log(Db.round(3).to_string(index=False)); log(json.dumps(Vd))


if __name__ == "__main__":
    {"lc_prepare": lc_prepare, "score": score}[sys.argv[1]]()
