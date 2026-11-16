"""Check 17 (amendment 13, CPU): held-out confirmation of pairwise cell-type calls, TMS.
    python c17_pairs.py prepare      # pair sample, LeafCutter inputs for the 50-pair subset -> $V/c17/ (array list lc/array_jobs.txt)
    python c17_pairs.py score        # LeafletFA and raw on all sampled pairs, LeafCutter on the subset, confirmation, rule
Calls from half-A TRAIN cells, confirmation on half-B TEST cells (TV > delta/2 and same sign at the lead junction), null = the
two types' labels permuted among their half-B TEST cells within mouse. Design 1 of check 16b (cap 5 per pair per bin)."""
import itertools, json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
import c10_catalog as C  # noqa: E402
from c10_catalog import Data, Groups, type_setup, mouse_halves, reads_cov, raw_pb, onehot, lead_sign, log  # noqa: E402
from c9_ds_null import lc_name  # noqa: E402

DS = "TMS"; OUT = os.path.join(vlib.V, "c17"); LCD = os.path.join(OUT, "lc")
DELTA = 0.10; CAP = 5; N_PERM = 20; PER_CLASS = 100; MIN_PAIRS = 10
BINS = [(1, 19, "1-19"), (20, 99, "20-99"), (100, 499, "100-499"), (500, np.inf, ">=500")]; COMPARED = ["20-99", "100-499", ">=500"]
LF, RAW, LC = "LeafletFA", "raw_pseudobulk", "LeafCutter"; BANDS = [(1, 5, "1-5"), (6, 20, "6-20"), (21, 100, "21-100")]


def slug(t):
    return "".join(c if c.isalnum() else "_" for c in t)


def sample_pairs(types, keep):
    names = sorted(types[k] for k in keep); code = {t: i for i, t in enumerate(types)}
    tis = {t: t.split("_", 1)[0] for t in names}; ct = {t: t.split("_", 1)[1] for t in names}
    cls = {"same_tissue": [], "same_cell_type": [], "unrelated": []}
    for a, b in itertools.combinations(names, 2):                                   # sorted order, a < b
        k = "same_tissue" if tis[a] == tis[b] else ("same_cell_type" if ct[a] == ct[b] else "unrelated")
        cls[k].append((a, b))
    rows = []
    for k, prs in cls.items():
        idx = np.sort(np.random.default_rng(17).choice(len(prs), min(PER_CLASS, len(prs)), replace=False))
        rows += [dict(pair_class=k, A=prs[i][0], B=prs[i][1], kA=code[prs[i][0]], kB=code[prs[i][1]], class_pairs=len(prs)) for i in idx]
    P = pd.DataFrame(rows); P["lc_subset"] = False
    for k, n in zip(cls, (17, 17, 16)):
        i = np.flatnonzero(P.pair_class == k)
        P.loc[i[np.sort(np.random.default_rng(1717).choice(len(i), min(n, len(i)), replace=False))], "lc_subset"] = True
    return P


def setup():
    D = Data(DS, 0); types, g = type_setup(D); G = len(types) + 1; inA = mouse_halves(D)
    R = pd.read_csv(os.path.join(vlib.V, "c10_lc", f"{DS}_halfA", "types.tsv"), sep="\t"); assert len(R) == 100
    assert all(types[c] == t for c, t in zip(R.code, R.type))
    return D, types, g, G, inA, R.code.values


def prepare():
    os.makedirs(LCD, exist_ok=True); D, types, g, G, inA, keep = setup()
    P = sample_pairs(types, keep); P.to_csv(os.path.join(OUT, "pairs.tsv"), sep="\t", index=False)
    log(P.groupby("pair_class").agg(sampled=("A", "size"), class_pairs=("class_pairs", "first"), lc_subset=("lc_subset", "sum")).to_string())
    ev = D.post["event_id"]; names = np.array([lc_name(j, ev[D.inv[i]]) for i, j in enumerate(D.post["junction_id"])])
    rows = np.flatnonzero(inA); mice = np.unique(D.mouse[rows]); mi = {m: i for i, m in enumerate(mice)}; mc = np.array([mi[m] for m in D.mouse[rows]])
    Y = D.Y[rows].tocsr(); jobs = []
    for x in P[P.lc_subset].itertuples():
        cols, grp, M = [], [], []
        for k, arm in ((x.kA, "armT"), (x.kB, "armR")):
            s = np.flatnonzero(g[rows] == k)
            Mo = sp.csr_matrix((np.ones(len(s)), (mc[s], np.arange(len(s)))), shape=(len(mice), len(s))); T = np.asarray((Mo @ Y[s]).todense())
            for i in np.flatnonzero(np.bincount(mc[s], minlength=len(mice)) > 0):
                cols.append(f"{mice[i]}_{arm}"); grp.append(arm); M.append(T[i])
        M = np.vstack(M).T; ok = M.sum(1) > 0; dd = os.path.join(LCD, f"{slug(x.A)}__{slug(x.B)}"); os.makedirs(dd, exist_ok=True)
        pd.DataFrame(M[ok].astype(int), index=names[ok], columns=cols).to_csv(os.path.join(dd, "counts.tsv.gz"), sep="\t", compression="gzip", index_label="chrom")
        open(os.path.join(dd, "groups.txt"), "w").write("".join(f"{c}\t{a}\n" for c, a in zip(cols, grp))); jobs.append(dd)
    open(os.path.join(LCD, "array_jobs.txt"), "w").write("\n".join(jobs) + "\n"); log(f"LeafCutter inputs: {len(jobs)} pairs")


def score():
    t0 = time.time(); D, types, g, G, inA, keep = setup(); D.load_test(DS)
    P = pd.read_csv(os.path.join(OUT, "pairs.tsv"), sep="\t"); pd.testing.assert_frame_equal(P, sample_pairs(types, keep), check_dtype=False)
    rA = np.flatnonzero(inA); Gr = Groups(D, rA, g[rA], G)
    S, _ = Gr.sums(D, D.post["phi"][rA], D.post["psi_mean"]); point = S / Gr.n[:, None]
    y, nn = raw_pb(D.Y, D.Ncell, D.inv, rA, g[rA], G)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.nan_to_num(y / nn)
    readsA, _ = reads_cov(D, rA, g[rA], G)
    _, _, tmouse = C.labels(DS, vlib.paths(DS)["test"]); te = D.test
    code = {t: i for i, t in enumerate(types)}; gt = np.array([code.get(t, len(types)) for t in te["t"]])
    mB = set(D.mouse[~inA]); inB = np.isin(tmouse, list(mB)); rB = np.flatnonzero(inB)
    Ob = onehot(gt[rB], G); readsB = np.asarray((Ob @ te["N"][rB]).todense()); covB = np.asarray((Ob @ (te["N"][rB] > 0).astype(np.float64)).todense())
    ev = D.post["event_id"]; jname = {lc_name(j, ev[D.inv[i]]): i for i, j in enumerate(D.post["junction_id"])}
    TY, TN = te["Y"].tocsr(), te["N"].tocsr(); out = []

    def pb(rows_, lab, k):
        r = rows_[lab == k]; yy = np.asarray(TY[r].sum(0)).ravel(); nb = np.asarray(TN[r].sum(0)).ravel()[D.inv]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nan_to_num(yy / nb)

    def conf(pa, pbb, E, lj, sg):
        d = pa - pbb; tvv = np.zeros(D.n_atse); np.add.at(tvv, D.inv, np.abs(d)); tvv *= 0.5
        return (tvv[E] > DELTA / 2) & (np.sign(d[np.maximum(lj, 0)]) == sg)
    for pi, x in enumerate(P.itertuples()):
        a, b = x.kA, x.kB
        E = np.flatnonzero((readsA[a] >= 1) & (readsA[b] >= 1) & (readsB[a] >= 20) & (readsB[b] >= 20) & (covB[a] >= 3) & (covB[b] >= 3))
        if not len(E):
            continue
        mr = np.minimum(readsA[a, E], readsA[b, E]); bn = np.select([mr <= 19, mr <= 99, mr <= 499], ["1-19", "20-99", "100-499"], ">=500")
        meth = {}
        for nm, X in ((LF, point), (RAW, raw)):
            lj, sg = lead_sign(X[[a]], X[[b]], D.inv, D.n_atse); lj, sg = lj[0, E], sg[0, E]
            meth[nm] = (np.abs(X[a, np.maximum(lj, 0)] - X[b, np.maximum(lj, 0)]), np.zeros(len(E)), lj, sg, np.ones(len(E), bool))
        if x.lc_subset:
            dd = os.path.join(LCD, f"{slug(x.A)}__{slug(x.B)}"); fs, fe = os.path.join(dd, "ds_cluster_significance.txt"), os.path.join(dd, "ds_effect_sizes.txt")
            st = np.full(len(E), np.nan); tie = np.zeros(len(E)); lj = np.full(len(E), -1); sg = np.zeros(len(E))
            if os.path.exists(fs) and os.path.exists(fe):
                c = pd.read_csv(fs, sep="\t"); c = c[c.status == "Success"]; e = pd.read_csv(fe, sep="\t"); e["j"] = e.intron.map(jname); assert e.j.notna().all()
                e["cl"] = e.intron.str.split(":").str[0] + ":" + e.intron.str.split(":").str[3]; e = e[e.cl.isin(set(c.cluster))]; e["a"] = D.inv[e.j.astype(int)]
                pc = dict(zip(c.cluster, c.p)); pos = {u: i for i, u in enumerate(E)}
                for at, yy in e.groupby("a"):
                    if at in pos:
                        l = yy.loc[yy.deltapsi_armT.abs().idxmax()]; i = pos[at]
                        st[i], tie[i], lj[i], sg[i] = -pc[l.cl], abs(l.deltapsi_armT), int(l.j), np.sign(l.deltapsi_armT)
            else:
                log(f"LeafCutter output missing: {dd}")
            meth[LC] = (st, tie, lj, sg, ~np.isnan(st))
        rows_ = rB[np.isin(gt[rB], [a, b])]; lab = gt[rows_]
        pa, pbb = pb(rows_, lab, a), pb(rows_, lab, b)
        nulls = []
        for p in range(N_PERM):
            rg = np.random.default_rng(1000 + p); lp = lab.copy()
            for m in np.unique(tmouse[rows_]):
                i = np.flatnonzero(tmouse[rows_] == m); lp[i] = rg.permutation(lab[i])
            nulls.append((pb(rows_, lp, a), pb(rows_, lp, b)))
        for nm, (st, tie, lj, sg, ok) in meth.items():
            real = conf(pa, pbb, E, lj, sg); nl = np.mean([conf(u, v, E, lj, sg) for u, v in nulls], axis=0)
            out.append(pd.DataFrame(dict(pair=pi, pair_class=x.pair_class, lc_subset=x.lc_subset, atse=E, bin=bn, method=nm, stat=st, tie=tie,
                                         scorable=ok, real=real.astype(float), null=nl)))
        if pi % 25 == 0:
            log(f"pair {pi}/{len(P)}")
    A = pd.concat(out, ignore_index=True); A["excess"] = A.real - A.null; A.to_parquet(os.path.join(OUT, "units.parquet"), index=False)

    def ordered(t):
        return t.iloc[np.lexsort((t.atse.values, -t.tie.values, -t.stat.values))]

    def boot(cm, pairs):
        pairs = np.array(sorted(pairs)); pos = {k: i for i, k in enumerate(pairs)}
        br = np.random.default_rng(0).integers(0, len(pairs), (2000, len(pairs))); w = np.stack([np.bincount(r, minlength=len(pairs)) for r in br])
        res = {}
        for m, t in cm.items():
            i = t.pair.map(pos).values; s_ = np.bincount(i, t.excess.values, len(pairs)); c_ = np.bincount(i, minlength=len(pairs)).astype(float)
            res[m] = (w @ s_) / np.maximum(w @ c_, 1e-12)
        return res
    summ, diffs = [], []
    for scope, cls in [("all", None)] + [(k, k) for k in ("same_tissue", "same_cell_type", "unrelated")]:
        for lab_ in [b_[2] for b_ in BINS]:
            for comp, sub_mask in ((RAW, None), (LC, "lc")):
                sub = A[(A.bin == lab_) & A.scorable & A.method.isin([LF, comp])]
                if cls:
                    sub = sub[sub.pair_class == cls]
                if sub_mask:
                    sub = sub[sub.lc_subset]
                n = sub.groupby(["pair", "method"]).size().unstack(1).reindex(columns=[LF, comp]).fillna(0)
                prs = n.index[(n >= CAP).all(1)].values
                cm = {m: ordered(sub[(sub.method == m) & sub.pair.isin(prs)]).groupby("pair", sort=False).head(CAP) for m in (LF, comp)}
                row = dict(scope=scope, bin=lab_, comparator=comp, pairs=len(prs), compared=len(prs) >= MIN_PAIRS)
                if len(prs):
                    bx = boot(cm, prs); dd = bx[LF] - bx[comp]
                    for m in (LF, comp):
                        summ.append(dict(scope=scope, bin=lab_, comparison=f"{LF} vs {comp}", method=m, pairs=len(prs), calls=len(cm[m]), real=cm[m].real.mean(),
                                         null=cm[m].null.mean(), excess=cm[m].excess.mean(), excess_lo=np.quantile(bx[m], .025), excess_hi=np.quantile(bx[m], .975)))
                    row.update(diff=cm[LF].excess.mean() - cm[comp].excess.mean(), diff_lo=np.quantile(dd, .025), diff_hi=np.quantile(dd, .975),
                               lf_excess_lo=np.quantile(bx[LF], .025))
                diffs.append(row)
    Sm, Df = pd.DataFrame(summ), pd.DataFrame(diffs)
    Sm.to_csv(os.path.join(OUT, "summary.tsv"), sep="\t", index=False, float_format="%.4f"); Df.to_csv(os.path.join(OUT, "differences.tsv"), sep="\t", index=False, float_format="%.4f")
    # rank bands (descriptive; the rate shown in the browser): rank within pair over all its units, LeafletFA and raw
    rb = []
    for m in (LF, RAW):
        t = A[(A.method == m) & A.scorable].copy(); t = t.iloc[np.lexsort((t.atse.values, -t.tie.values, -t.stat.values, t.pair.values))]
        t["rank"] = t.groupby("pair").cumcount() + 1
        for lo, hi, bl in BANDS:
            for lab_, u in t[(t["rank"] >= lo) & (t["rank"] <= hi)].groupby("bin"):
                rb.append(dict(method=m, rank_band=bl, bin=lab_, calls=len(u), pairs=u.pair.nunique(), confirmed=u.real.mean(), null=u.null.mean(), excess=u.excess.mean()))
    pd.DataFrame(rb).to_csv(os.path.join(OUT, "rank_bands.tsv"), sep="\t", index=False, float_format="%.4f")
    Pm = Df[(Df.scope == "all") & Df.bin.isin(COMPARED) & Df.compared]
    noninf = bool(len(Pm) and (Pm.diff_lo > -0.05).all())
    better = noninf and all((Pm[Pm.comparator == c].diff_lo > 0).any() for c in (RAW, LC) if (Pm.comparator == c).any())
    th = Df[(Df.scope == "all") & (Df.bin == "1-19") & (Df.comparator == RAW)]
    thin = bool(len(th) and th.compared.iloc[0] and th.lf_excess_lo.iloc[0] > 0 and th.diff_lo.iloc[0] > 0)
    V = dict(pairs=len(P), pairs_with_units=int(A.pair.nunique()), lc_pairs=int(P.lc_subset.sum()),
             lc_outputs=int(sum(os.path.exists(os.path.join(LCD, f"{slug(x.A)}__{slug(x.B)}", "ds_cluster_significance.txt")) for x in P[P.lc_subset].itertuples())),
             compared=[f"{r.bin}:{r.comparator}" for r in Pm.itertuples()], not_compared=[f"{r.bin}:{r.comparator} ({r.pairs} pairs)" for r in
             Df[(Df.scope == "all") & Df.bin.isin(COMPARED) & ~Df.compared].itertuples()],
             non_inferior=noninf, better=better, verdict="BETTER" if better else ("NON-INFERIOR" if noninf else "FAIL"), thin_1_19_supported=thin,
             minutes=(time.time() - t0) / 60)
    json.dump(V, open(os.path.join(OUT, "verdict.json"), "w"), indent=1)
    log(Df[Df.scope == "all"].round(3).to_string(index=False)); log(json.dumps(V))


if __name__ == "__main__":
    {"prepare": prepare, "score": score}[sys.argv[1]]()
