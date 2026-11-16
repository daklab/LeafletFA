"""Check 16 (amendment 12, CPU): held-out confirmation of type-vs-rest calls at a matched number of calls, TMS.
    python c16_ds_heldout.py            # -> $V/c16/ (C16_OUT, C16_DRAWS, C16_PERM override for the smoke run)
Calls from half-A TRAIN cells (LeafletFA z, raw pseudobulk |dpsi|, LeafCutter p from the existing c10_lc/TMS_halfA runs);
confirmation on half-B TEST cells only (10.2 (ii): TV > delta/2 and same sign at the call's lead junction); null = half-B
TEST labels permuted within mouse. Units, bins, N_b and the pass rule are those of VALIDATION_PREREG.md amendment 12."""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
import c10_catalog as C  # noqa: E402
from c10_catalog import Data, Groups, type_setup, mouse_halves, reads_cov, raw_pb, onehot, tv, lead_sign, phi_draw, psi_draw, log  # noqa: E402
from c9_ds_null import lc_name  # noqa: E402

DS = "TMS"; OUT = os.environ.get("C16_OUT", os.path.join(vlib.V, "c16"))
S_DRAWS = int(os.environ.get("C16_DRAWS", 200)); N_PERM = int(os.environ.get("C16_PERM", 20))
DELTA = json.load(open(os.path.join(vlib.V, "tsv", f"c10_{DS}_s0_delta.json")))["delta"]
TAU = pd.read_csv(os.path.join(vlib.V, "tsv", f"c12_{DS}_tau2.tsv"), sep="\t")
TAU = TAU[(TAU.draws == "combined") & (TAU.calibrated_on == "A")].set_index("test_read_bin").tau2
TAU_LO, TAU_HI = float(TAU["100-499"]), float(TAU["500-inf"])
BINS = [(1, 19, "1-19"), (20, 99, "20-99"), (100, 499, "100-499"), (500, np.inf, ">=500")]
COMPARED = ["20-99", "100-499", ">=500"]; NB = 2000; SEED_DRAW = 101
LC_DIR = os.path.join(vlib.V, "c10_lc", f"{DS}_halfA")


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    assert DELTA == 0.10 and TAU_LO == 0.01024 and TAU_HI == 0.00256, (DELTA, TAU_LO, TAU_HI)
    D = Data(DS, 0); D.load_test(DS); types, g = type_setup(D); G = len(types) + 1; inA = mouse_halves(D); rA = np.flatnonzero(inA)
    R = pd.read_csv(os.path.join(LC_DIR, "types.tsv"), sep="\t"); assert len(R) == 100
    assert all(types[c] == t for c, t in zip(R.code, R.type)); keep = R.code.values
    # ---- LeafletFA, half-A TRAIN: point, lead, draw variance of dpsi (combined draws, as estimates() with seed 101) ----
    Gr = Groups(D, rA, g[rA], G); n = Gr.n; tot_n = n.sum()
    S, _ = Gr.sums(D, D.post["phi"][rA], D.post["psi_mean"])
    point = S / n[:, None]; rest = (S.sum(0)[None, :] - S) / (tot_n - n)[:, None]
    ljL, sgL = lead_sign(point[:-1], rest[:-1], D.inv, D.n_atse)
    dL = np.take_along_axis(point[:-1] - rest[:-1], np.maximum(ljL, 0), axis=1)
    gen = torch.Generator().manual_seed(SEED_DRAW); rng = np.random.default_rng(SEED_DRAW + 7)
    s1 = np.zeros((G - 1, D.n_atse)); s2 = np.zeros((G - 1, D.n_atse))
    for s in range(S_DRAWS):
        ph, ps = phi_draw(D, gen)[rA], psi_draw(D, gen)
        Sb, Nb = Gr.sums(D, ph, ps, Gr.boot_weights(rng))
        x = Sb[:-1] / np.maximum(Nb[:-1], 1e-12)[:, None]; r = (Sb.sum(0)[None, :] - Sb[:-1]) / (Nb.sum() - Nb[:-1])[:, None]
        d = np.take_along_axis((x - r).astype(np.float64), np.maximum(ljL, 0), axis=1); s1 += d; s2 += d * d
        if s % 25 == 0:
            log(f"draw {s}/{S_DRAWS}")
    var = np.maximum(s2 / S_DRAWS - (s1 / S_DRAWS) ** 2, 0) * S_DRAWS / max(S_DRAWS - 1, 1)
    readsA, covA = reads_cov(D, rA, g[rA], G); readsA, covA = readsA[:-1], covA[:-1]
    tau2 = np.where(readsA < 500, TAU_LO, TAU_HI)
    zL = np.abs(dL) / np.sqrt(var + tau2)
    # ---- raw pseudobulk, half-A TRAIN ----
    y, nn = raw_pb(D.Y, D.Ncell, D.inv, rA, g[rA], G)
    with np.errstate(invalid="ignore", divide="ignore"):
        rp = np.nan_to_num(y[:-1] / nn[:-1]); rr = np.nan_to_num((y.sum(0) - y[:-1]) / (nn.sum(0) - nn[:-1]))
    ljP, sgP = lead_sign(rp, rr, D.inv, D.n_atse); dP = np.abs(np.take_along_axis(rp - rr, np.maximum(ljP, 0), axis=1))
    # ---- half-B TEST cells ----
    _, _, tmouse = C.labels(DS, vlib.paths(DS)["test"]); te = D.test
    code = {t: i for i, t in enumerate(types)}; gt = np.array([code.get(t, len(types)) for t in te["t"]])
    mA = set(D.mouse[inA]); mB = set(D.mouse[~inA]); inB = np.isin(tmouse, list(mB)); rB = np.flatnonzero(inB)
    assert not (np.isin(tmouse, list(mA)) & inB).any()
    log(f"half-B TEST cells {len(rB):,} of {len(gt):,}; mice B {len(mB)}")
    Ob = onehot(gt[rB], G); readsB = np.asarray((Ob @ te["N"][rB]).todense())[:-1]; covB = np.asarray((Ob @ (te["N"][rB] > 0).astype(np.float64)).todense())[:-1]

    def bstats(lab):
        yb, nb = raw_pb(te["Y"], te["N"], D.inv, rB, lab[rB], G)
        with np.errstate(invalid="ignore", divide="ignore"):
            a = np.nan_to_num(yb[:-1] / nb[:-1]); b = np.nan_to_num((yb.sum(0) - yb[:-1]) / (nb.sum(0) - nb[:-1]))
        return tv(a, b, D.inv, D.n_atse), a - b
    # ---- LeafCutter: existing half-A runs ----
    ev = D.post["event_id"]; jname = {lc_name(j, ev[D.inv[i]]): i for i, j in enumerate(D.post["junction_id"])}
    pLC = np.full((G - 1, D.n_atse), np.nan); ljC = np.full((G - 1, D.n_atse), -1); sgC = np.zeros((G - 1, D.n_atse)); dC = np.zeros((G - 1, D.n_atse))
    for x in R.itertuples():
        c = pd.read_csv(os.path.join(x.dir, "ds_cluster_significance.txt"), sep="\t"); c = c[c.status == "Success"]
        e = pd.read_csv(os.path.join(x.dir, "ds_effect_sizes.txt"), sep="\t"); e["j"] = e.intron.map(jname); assert e.j.notna().all()
        e["cl"] = e.intron.str.split(":").str[0] + ":" + e.intron.str.split(":").str[3]; e = e[e.cl.isin(set(c.cluster))]
        e["a"] = D.inv[e.j.astype(int)]; pc = dict(zip(c.cluster, c.p))
        for a, yy in e.groupby("a"):
            l = yy.loc[yy.deltapsi_armT.abs().idxmax()]
            pLC[x.code, a] = pc[l.cl]; ljC[x.code, a] = int(l.j); sgC[x.code, a] = np.sign(l.deltapsi_armT); dC[x.code, a] = abs(l.deltapsi_armT)
    # ---- universe U ----
    K, E = np.nonzero((readsA >= 1) & (readsB >= 20) & (covB >= 3))
    m = np.isin(K, keep); K, E = K[m], E[m]                                       # row-major order = (type code, ATSE) tie order
    U = pd.DataFrame(dict(k=K, a=E, readsA=readsA[K, E], readsB=readsB[K, E]))
    U["bin"] = ""
    for lo, hi, lab in BINS:
        U.loc[(U.readsA >= lo) & (U.readsA <= hi), "bin"] = lab
    meth = {"LeafletFA_z": (zL[K, E], ljL[K, E], sgL[K, E], np.ones(len(K), bool)),
            "raw_pseudobulk": (dP[K, E], ljP[K, E], sgP[K, E], np.ones(len(K), bool)),
            "LeafCutter": (-pLC[K, E], ljC[K, E], sgC[K, E], ~np.isnan(pLC[K, E])),
            "LeafletFA_point_descriptive": (np.abs(dL[K, E]), ljL[K, E], sgL[K, E], np.ones(len(K), bool))}
    tie2 = {"LeafCutter": dC[K, E]}
    # ---- confirmation: real and within-mouse permutations of half-B TEST labels ----
    tvR, dR = bstats(gt)
    nulls = []
    for p in range(N_PERM):
        rg = np.random.default_rng(1000 + p); gp = gt.copy()
        for mm in np.unique(tmouse[rB]):
            i = np.flatnonzero(inB & (tmouse == mm)); gp[i] = rg.permutation(gt[i])
        nulls.append(bstats(gp)); log(f"null {p}")

    def conf(tvX, dX, lj, sg):
        return (tvX[K, E] > DELTA / 2) & (np.sign(dX[K, np.maximum(lj, 0)]) == sg)
    rows_all = []
    for name, (stat, lj, sg, ok) in meth.items():
        real = conf(tvR, dR, lj, sg); sign_only = np.sign(dR[K, np.maximum(lj, 0)]) == sg
        nul = np.mean([conf(a_, b_, lj, sg) for a_, b_ in nulls], axis=0)
        rows_all.append(pd.DataFrame(dict(k=K, a=E, method=name, stat=stat, tie=tie2.get(name, np.zeros(len(K))), scorable=ok,
                                          real=real.astype(float), null=nul, sign=sign_only.astype(float), bin=U.bin.values, lead=lj)))
    A = pd.concat(rows_all, ignore_index=True)
    if os.environ.get("C16_DUMP"):                                              # amendment 12 note 1 (check 16b): every unit, every method
        A.to_parquet(os.path.join(OUT, "units.parquet"), index=False)

    def top(sub, N):
        sub = sub[sub.scorable]
        o = np.lexsort((np.arange(len(sub)), -sub.tie.values, -sub.stat.values))   # stat desc, tie desc, (k, a) order
        return sub.iloc[o[:N]]
    calls = []
    for lab in [b[2] for b in BINS]:
        ub = int((U.bin == lab).sum())
        for design, N in (("primary", min(500, ub // 10)), ("sensitivity", min(2000, ub // 4))):
            for name in meth:
                sub = A[(A.method == name) & (A.bin == lab)]; t_ = top(sub, N)
                calls.append(t_.assign(design=design, N=N, U_bin=ub, n_scorable=int(sub.scorable.sum())))
    for name in meth:
        t_ = top(A[A.method == name], NB); calls.append(t_.assign(design="pooled_descriptive", bin="all", N=NB, U_bin=len(U), n_scorable=int(A[A.method == name].scorable.sum())))
    Cc = pd.concat(calls, ignore_index=True); Cc["excess"] = Cc.real - Cc.null; Cc["cell_type"] = np.array(types)[Cc.k]
    Cc.to_csv(os.path.join(OUT, "calls.tsv.gz"), sep="\t", index=False, float_format="%.5g")
    # ---- type bootstrap (2,000, seed 0): each method's excess on its own calls; differences within a resample ----
    br = np.random.default_rng(0).integers(0, len(keep), (2000, len(keep))); wts = np.stack([np.bincount(b, minlength=len(keep)) for b in br])
    kpos = {k: i for i, k in enumerate(keep)}

    def bvals(t_, col):
        i = t_.k.map(kpos).values; sm = np.bincount(i, t_[col].values, len(keep)); ct = np.bincount(i, minlength=len(keep)).astype(float)
        return (wts @ sm) / np.maximum(wts @ ct, 1e-12)
    summ = []; boots = {}
    for (design, lab, name), t_ in Cc.groupby(["design", "bin", "method"]):
        N, nsc = int(t_.N.iloc[0]), int(t_.n_scorable.iloc[0])
        bx = bvals(t_, "excess"); boots[(design, lab, name)] = bx
        summ.append(dict(design=design, bin=lab, method=name, N=N, U_bin=int(t_.U_bin.iloc[0]), scorable=nsc, calls=len(t_), compared=nsc >= N,
                         real=t_.real.mean(), null=t_.null.mean(), excess=t_.excess.mean(), excess_lo=np.quantile(bx, .025),
                         excess_hi=np.quantile(bx, .975), sign_agreement=t_.sign.mean(), types_with_calls=t_.k.nunique()))
    Sm = pd.DataFrame(summ); Sm.to_csv(os.path.join(OUT, "summary.tsv"), sep="\t", index=False, float_format="%.4f")
    diffs = []
    for design in ("primary", "sensitivity", "pooled_descriptive"):
        for lab in sorted(set(Cc[Cc.design == design].bin)):
            rL = Sm[(Sm.design == design) & (Sm.bin == lab) & (Sm.method == "LeafletFA_z")].iloc[0]
            for comp in ("raw_pseudobulk", "LeafCutter", "LeafletFA_point_descriptive"):
                rC = Sm[(Sm.design == design) & (Sm.bin == lab) & (Sm.method == comp)]
                if rC.empty:                                                    # no scorable unit in this bin (LeafCutter at 1-19)
                    diffs.append(dict(design=design, bin=lab, comparator=comp, both_compared=False, diff=np.nan, diff_lo=np.nan, diff_hi=np.nan))
                    continue
                rC = rC.iloc[0]
                dd = boots[(design, lab, "LeafletFA_z")] - boots[(design, lab, comp)]
                diffs.append(dict(design=design, bin=lab, comparator=comp, both_compared=bool(rL.compared and rC.compared),
                                  diff=rL.excess - rC.excess, diff_lo=float(np.quantile(dd, .025)), diff_hi=float(np.quantile(dd, .975))))
    Df = pd.DataFrame(diffs)
    Df.to_csv(os.path.join(OUT, "differences.tsv"), sep="\t", index=False, float_format="%.4f")
    # ---- pass rule (amendment 12) ----
    P = Df[(Df.design == "primary") & Df.bin.isin(COMPARED) & Df.comparator.isin(["raw_pseudobulk", "LeafCutter"]) & Df.both_compared]
    noninf = bool(len(P) and (P.diff_lo > -0.05).all())
    better = noninf and all((P[P.comparator == c].diff_lo > 0).any() for c in ("raw_pseudobulk", "LeafCutter"))
    thinL = Sm[(Sm.design == "primary") & (Sm.bin == "1-19") & (Sm.method == "LeafletFA_z")]
    thinD = Df[(Df.design == "primary") & (Df.bin == "1-19") & (Df.comparator == "raw_pseudobulk")]
    thin = bool(len(thinL) and thinL.excess_lo.iloc[0] > 0 and len(thinD) and thinD.diff_lo.iloc[0] > 0)
    verdict = dict(dataset=DS, delta=DELTA, tau2_A=[TAU_LO, TAU_HI], draws=S_DRAWS, perms=N_PERM, U=len(U),
                   U_by_bin=U.bin.value_counts().to_dict(), comparisons=len(P), comparisons_skipped=[f"{r.bin}:{r.comparator}" for r in
                   Df[(Df.design == "primary") & Df.bin.isin(COMPARED) & Df.comparator.isin(["raw_pseudobulk", "LeafCutter"]) & ~Df.both_compared].itertuples()],
                   non_inferior=noninf, better=better, verdict="BETTER" if better else ("NON-INFERIOR" if noninf else "FAIL"),
                   thin_1_19_supported=thin, half_B_test_cells=int(len(rB)), minutes=(time.time() - t0) / 60)
    json.dump(verdict, open(os.path.join(OUT, "verdict.json"), "w"), indent=1, default=str)
    log(Sm[Sm.design == "primary"].round(3).to_string(index=False)); log(Df[Df.design == "primary"].round(3).to_string(index=False)); log(json.dumps(verdict, default=str))


if __name__ == "__main__":
    main()
