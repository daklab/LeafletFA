"""Check 19 (amendment 15, CPU): Allen pairs.
    python c19_allen_pairs.py export   # part 1: all 2,278 pairs, top 200 ATSEs by |dpsi_lead|, donors_both, flag (descriptive)
    python c19_allen_pairs.py score    # part 2: within-donor confirmation on pairs with >= 4 shared donors (check 17 rule)
C19_MAXPAIRS=n limits part 2 to the first n eligible pairs (smoke run; output to $V/c19_smoke)."""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
import c10_catalog as C  # noqa: E402
from c10_catalog import Data, Groups, type_setup, onehot, log  # noqa: E402

MAXP = int(os.environ.get("C19_MAXPAIRS", 0)); OUT = os.path.join(vlib.V, "c19_smoke" if MAXP else "c19")
DELTA = 0.10; CAP = 5; N_PERM = 20; MIN_UNITS = 10; TOPN = 200
BINS = ["1-19", "20-99", "100-499", ">=500"]; COMPARED = ["20-99", "100-499", ">=500"]
LF, RAW = "LeafletFA", "raw_pseudobulk"; BANDS = [(1, 5, "1-5"), (6, 20, "6-20"), (21, 100, "21-100")]
FLAG = "cell type and donor not separable; differences may reflect donor or batch"
META = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/METADATA/metadata.csv"


class Lead:
    """vectorized per-ATSE argmax |d| over junctions (first junction in column order on ties, as lead_sign)"""
    def __init__(self, inv):
        self.o = np.argsort(inv, kind="stable"); s = inv[self.o]; self.st = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
        self.n_atse = len(self.st); self.J = len(inv)

    def __call__(self, d):                                            # d: (P, J) -> lead junction (P, n_atse), sign
        a = np.abs(np.nan_to_num(d[:, self.o], nan=-1.0)); mx = np.maximum.reduceat(a, self.st, axis=1)
        seg = np.repeat(np.arange(self.n_atse), np.diff(np.r_[self.st, self.J]))
        idx = np.where(a == mx[:, seg], self.o[None, :], self.J)
        lj = np.minimum.reduceat(idx, self.st, axis=1)
        return lj, np.sign(np.take_along_axis(np.nan_to_num(d), lj, axis=1))


def tv(d, inv, n_atse):
    out = np.zeros(d.shape[:-1] + (n_atse,)); np.add.at(out.T, inv, np.abs(np.nan_to_num(d)).T); return 0.5 * out


def donors_table():
    T = pd.read_csv(os.path.join(vlib.V, "c17", "allen_donors_per_pair.tsv"), sep="\t"); return T


def group_class(groups):
    S = pd.read_csv(os.path.join(vlib.V, "tsv", "c10_Allen_group_subclass.tsv"), sep="\t").set_index("cell_group")
    m = pd.read_csv(META, usecols=["subclass_label", "class_label"], low_memory=False).drop_duplicates()
    assert m.subclass_label.is_unique, "a subclass maps to more than one class"
    s2c = dict(zip(m.subclass_label, m.class_label))
    return {g: s2c[S.loc[g, "official_subclass"]] for g in groups}


# ---------------- part 1 ----------------
def export():
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    X = pd.read_csv(os.path.join(vlib.V, "c12", "export", "catalog_Allen_s0.tsv.gz"), sep="\t",
                    usecols=["event_id", "junction_id", "gene", "cell_type", "psi", "raw_psi", "atse_reads", "cells_covering"])
    groups = sorted(X.cell_type.unique()); assert len(groups) == 68
    jun = X.drop_duplicates("junction_id")[["junction_id", "event_id", "gene"]].reset_index(drop=True)
    jpos = pd.Series(np.arange(len(jun)), index=jun.junction_id); ev = jun.event_id.values; _, inv = np.unique(ev, return_inverse=True)
    evn = np.unique(ev); gene_e = pd.Series(jun.gene.values, index=ev).groupby(level=0).first().reindex(evn).values
    gpos = {g: i for i, g in enumerate(groups)}; J, G, NE = len(jun), len(groups), len(evn)
    psi = np.full((G, J), np.nan); raw = np.full((G, J), np.nan); rd = np.zeros((G, NE)); cv = np.zeros((G, NE))
    gi = X.cell_type.map(gpos).values; ji = jpos[X.junction_id].values
    psi[gi, ji] = X.psi.values; raw[gi, ji] = X.raw_psi.values; rd[gi, inv[ji]] = X.atse_reads.values; cv[gi, inv[ji]] = X.cells_covering.values
    del X
    D = donors_table().set_index(["A", "B"]); lead = Lead(inv); rows, idx = [], []
    for i, a in enumerate(groups):
        bs = groups[i + 1:]
        if not bs:
            continue
        b_ = np.array([gpos[b] for b in bs]); d = psi[[gpos[a]]] - psi[b_]
        lj, sg = lead(d)
        for r, b in enumerate(bs):
            k = gpos[b]; ok = (rd[gpos[a]] >= 20) & (rd[k] >= 20) & (cv[gpos[a]] >= 3) & (cv[k] >= 3)
            e = np.flatnonzero(ok); j = lj[r, e]; dd = d[r, j]; top = e[np.argsort(-np.abs(dd), kind="stable")[:TOPN]]; jt = lj[r, top]
            nd = int(D.loc[(a, b), "donors_both_train"])
            idx.append(dict(A=a, B=b, donors_both=nd, flag=FLAG if nd == 0 else "", eligible_atses=len(e)))
            rows.append(pd.DataFrame(dict(A=a, B=b, rank=np.arange(1, len(top) + 1), gene=gene_e[top], event_id=evn[top], lead_junction=jun.junction_id.values[jt],
                                          psi_A=psi[gpos[a], jt], psi_B=psi[k, jt], delta_psi=psi[gpos[a], jt] - psi[k, jt], raw_psi_A=raw[gpos[a], jt], raw_psi_B=raw[k, jt],
                                          atse_reads_A=rd[gpos[a], top], atse_reads_B=rd[k, top], cells_covering_A=cv[gpos[a], top], cells_covering_B=cv[k, top], donors_both=nd)))
    I = pd.DataFrame(idx); I.to_csv(os.path.join(OUT, "allen_pairs_index.tsv"), sep="\t", index=False)
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(OUT, "allen_pairs_top200.tsv.gz"), sep="\t", index=False, float_format="%.4g")
    log(f"{len(I)} pairs; flagged (0 donors) {int((I.donors_both == 0).sum())}; median eligible ATSEs {I.eligible_atses.median():.0f}; {(time.time() - t0) / 60:.1f} min")


# ---------------- part 2 ----------------
def score():
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    D = Data("Allen", 0); D.load_test("Allen"); types, g = type_setup(D); assert len(types) == 68
    _, _, tmouse = C.labels("Allen", vlib.paths("Allen")["test"]); te = D.test
    code = {t: i for i, t in enumerate(types)}; gt = np.array([code.get(t, -1) for t in te["t"]])
    T = donors_table(); T = T[T.donors_both_train >= 4].reset_index(drop=True); assert len(T) == 642
    if MAXP:
        T = T.iloc[:MAXP]
    cls = group_class(types)
    # shared donors per pair, from TRAIN cells
    dons = pd.DataFrame(dict(k=g, m=D.mouse)).query("k < 68").groupby("k").m.agg(lambda s: set(s)).to_dict()
    shared = [sorted(dons[code[r.A]] & dons[code[r.B]]) for r in T.itertuples()]
    assert all(len(s) == r.donors_both_train for s, r in zip(shared, T.itertuples()))
    combos = sorted({(code[t], d) for r, s in zip(T.itertuples(), shared) for t in (r.A, r.B) for d in s}); cpos = {c: i for i, c in enumerate(combos)}; Gc = len(combos)
    log(f"{len(T)} pairs, {Gc} (group, donor) combos")
    key_tr = pd.Series(list(zip(g, D.mouse))).map(cpos); rtr = np.flatnonzero(key_tr.notna().values); ctr = key_tr.values[rtr].astype(int)
    Gr = Groups(D, rtr, ctr, Gc); S, _ = Gr.sums(D, D.post["phi"][rtr], D.post["psi_mean"]); lfp = (S / Gr.n[:, None]).astype(np.float32); del S, Gr
    O = onehot(ctr, Gc); ytr = np.asarray((O @ D.Y[rtr]).todense(), np.float32); ntr = np.asarray((O @ D.Ncell[rtr]).todense(), np.float32)
    key_te = pd.Series(list(zip(gt, tmouse))).map(cpos); rte = np.flatnonzero(key_te.notna().values); cte = key_te.values[rte].astype(int)
    Ot = onehot(cte, Gc); yte = np.asarray((Ot @ te["Y"][rte]).todense(), np.float32); nte = np.asarray((Ot @ te["N"][rte]).todense(), np.float32)
    cvte = np.asarray((Ot @ (te["N"][rte] > 0).astype(np.float64)).todense(), np.float32)
    TY, TN = te["Y"].tocsr(), te["N"].tocsr(); inv = D.inv; lead = Lead(inv); out = []

    def wmean(pa, pb, na, nb):
        """equal-weight mean over donors with >= 1 read in both groups at the ATSE of per-donor (pa - pb); (J,), and any-donor mask (n_atse,)"""
        ok = (na >= 1) & (nb >= 1); W = ok[:, inv].astype(np.float64); s = W.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(s > 0, (W * np.nan_to_num(pa - pb)).sum(0) / s, np.nan), ok.any(0)

    def psi_of(y, n):
        with np.errstate(invalid="ignore", divide="ignore"):
            return y / n[:, inv]
    for pi, (r, ds) in enumerate(zip(T.itertuples(), shared)):
        a, b = code[r.A], code[r.B]; ia = [cpos[(a, d)] for d in ds]; ib = [cpos[(b, d)] for d in ds]
        nA, nB = ntr[ia], ntr[ib]
        dLF, okt = wmean(lfp[ia], lfp[ib], nA, nB); dRW, _ = wmean(psi_of(ytr[ia], nA), psi_of(ytr[ib], nB), nA, nB)
        rA, rB = nA.sum(0), nB.sum(0)
        E = np.flatnonzero(okt & (nte[ia].sum(0) >= 20) & (nte[ib].sum(0) >= 20) & (cvte[ia].sum(0) >= 3) & (cvte[ib].sum(0) >= 3))
        if not len(E):
            continue
        mr = np.minimum(rA[E], rB[E]); bn = np.select([mr <= 19, mr <= 99, mr <= 499], ["1-19", "20-99", "100-499"], ">=500")
        dT, _ = wmean(psi_of(yte[ia], nte[ia]), psi_of(yte[ib], nte[ib]), nte[ia], nte[ib])
        # null: labels permuted among the pair's TEST cells within each donor
        rows_ = rte[np.isin(cte, ia + ib)]; lab = gt[rows_]; dm = tmouse[rows_]; dpos = {d: i for i, d in enumerate(ds)}; di = np.array([dpos[x] for x in dm])
        nulls = []
        for p in range(N_PERM):
            rg = np.random.default_rng(1000 + p); lp = lab.copy()
            for d in range(len(ds)):
                i = np.flatnonzero(di == d); lp[i] = rg.permutation(lab[i])
            key = np.where(lp == a, 0, 1) * len(ds) + di; Op = onehot(key, 2 * len(ds))
            yp = np.asarray((Op @ TY[rows_]).todense()); np_ = np.asarray((Op @ TN[rows_]).todense())
            nulls.append(wmean(psi_of(yp[:len(ds)], np_[:len(ds)]), psi_of(yp[len(ds):], np_[len(ds):]), np_[:len(ds)], np_[len(ds):])[0])
        tvT = tv(dT[None], inv, D.n_atse)[0]; tvN = [tv(u[None], inv, D.n_atse)[0] for u in nulls]
        for nm, dd in ((LF, dLF), (RAW, dRW)):
            lj, sg = lead(dd[None]); lj, sg = lj[0, E], sg[0, E]; st = np.abs(dd[lj])
            real = (tvT[E] > DELTA / 2) & (np.sign(np.nan_to_num(dT[lj])) == sg)
            nl = np.mean([(tn[E] > DELTA / 2) & (np.sign(np.nan_to_num(u[lj])) == sg) for tn, u in zip(tvN, nulls)], axis=0)
            out.append(pd.DataFrame(dict(grp=pi, kind="same_class" if cls[r.A] == cls[r.B] else "different_class", atse=E, bin=bn, method=nm,
                                         stat=st, tie=0.0, scorable=~np.isnan(st), real=real.astype(float), null=nl)))
        if pi % 50 == 0:
            log(f"pair {pi}/{len(T)}")
    U = pd.concat(out, ignore_index=True); U["excess"] = U.real - U.null; U.to_parquet(os.path.join(OUT, "units.parquet"), index=False)

    def ordered(t):
        return t.iloc[np.lexsort((t.atse.values, -t.tie.values, -t.stat.values))]

    def boot(cm, us):
        us = np.array(sorted(us)); pos = {k: i for i, k in enumerate(us)}
        br = np.random.default_rng(0).integers(0, len(us), (2000, len(us))); w = np.stack([np.bincount(x, minlength=len(us)) for x in br]); res = {}
        for m, t in cm.items():
            i = t.grp.map(pos).values; s_ = np.bincount(i, t.excess.values, len(us)); c_ = np.bincount(i, minlength=len(us)).astype(float)
            res[m] = (w @ s_) / np.maximum(w @ c_, 1e-12)
        return res
    summ, diffs = [], []
    for scope in ("all", "same_class", "different_class"):
        for b in BINS:
            sub = U[(U.bin == b) & U.scorable]
            if scope != "all":
                sub = sub[sub.kind == scope]
            n = sub.groupby(["grp", "method"]).size().unstack(1).reindex(columns=[LF, RAW]).fillna(0); us = n.index[(n >= CAP).all(1)].values
            cm = {m: ordered(sub[(sub.method == m) & sub.grp.isin(us)]).groupby("grp", sort=False).head(CAP) for m in (LF, RAW)}
            row = dict(scope=scope, bin=b, comparator=RAW, pairs=len(us), compared=len(us) >= MIN_UNITS)
            if len(us):
                bx = boot(cm, us); dd = bx[LF] - bx[RAW]
                for m in (LF, RAW):
                    summ.append(dict(scope=scope, bin=b, method=m, pairs=len(us), calls=len(cm[m]), real=cm[m].real.mean(), null=cm[m].null.mean(),
                                     excess=cm[m].excess.mean(), excess_lo=np.quantile(bx[m], .025), excess_hi=np.quantile(bx[m], .975)))
                row.update(diff=cm[LF].excess.mean() - cm[RAW].excess.mean(), diff_lo=np.quantile(dd, .025), diff_hi=np.quantile(dd, .975), lf_excess_lo=np.quantile(bx[LF], .025))
            diffs.append(row)
    Sm, Df = pd.DataFrame(summ), pd.DataFrame(diffs)
    Sm.to_csv(os.path.join(OUT, "summary.tsv"), sep="\t", index=False, float_format="%.4f"); Df.to_csv(os.path.join(OUT, "differences.tsv"), sep="\t", index=False, float_format="%.4f")
    rb = []
    for m in (LF, RAW):
        t = U[(U.method == m) & U.scorable].copy(); t = t.iloc[np.lexsort((t.atse.values, -t.tie.values, -t.stat.values, t.grp.values))]; t["rank"] = t.groupby("grp").cumcount() + 1
        for lo, hi, bl in BANDS:
            for b, u in t[(t["rank"] >= lo) & (t["rank"] <= hi)].groupby("bin"):
                rb.append(dict(method=m, rank_band=bl, bin=b, calls=len(u), pairs=u.grp.nunique(), confirmed=u.real.mean(), null=u.null.mean(), excess=u.excess.mean()))
    pd.DataFrame(rb).to_csv(os.path.join(OUT, "rank_bands.tsv"), sep="\t", index=False, float_format="%.4f")
    Pm = Df[(Df.scope == "all") & Df.bin.isin(COMPARED) & Df.compared]
    noninf = bool(len(Pm) and (Pm.diff_lo > -0.05).all()); better = noninf and bool((Pm.diff_lo > 0).any())
    th = Df[(Df.scope == "all") & (Df.bin == "1-19")]
    thin = bool(len(th) and th.compared.iloc[0] and th.lf_excess_lo.iloc[0] > 0 and th.diff_lo.iloc[0] > 0)
    sd = pd.cut(T.donors_both_train, [3, 5, 9, 19, np.inf], labels=["4-5", "6-9", "10-19", ">=20"]).value_counts().sort_index().to_dict()
    Vd = dict(pairs=len(T), pairs_with_units=int(U.grp.nunique()), kinds=pd.Series([("same_class" if cls[r.A] == cls[r.B] else "different_class") for r in T.itertuples()]).value_counts().to_dict(),
              shared_donors={str(k): int(v) for k, v in sd.items()}, compared=[f"{r.bin} ({r.pairs})" for r in Pm.itertuples()],
              not_compared=[f"{r.bin} ({r.pairs})" for r in Df[(Df.scope == "all") & Df.bin.isin(COMPARED) & ~Df.compared].itertuples()],
              non_inferior=noninf, better=better, verdict="BETTER" if better else ("NON-INFERIOR" if noninf else "FAIL"), thin_1_19_supported=thin,
              combos=Gc, minutes=(time.time() - t0) / 60)
    json.dump(Vd, open(os.path.join(OUT, "verdict.json"), "w"), indent=1)
    log(Df.round(3).to_string(index=False)); log(json.dumps(Vd))


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1]]()
