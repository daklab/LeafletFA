"""Check 16b (amendment 12 note 1, CPU): are check 16's differences explained by concentration of calls in few cell types?
    C16_DUMP=1 C16_OUT=$V/c16b python c16_ds_heldout.py   # regenerates check 16 with the per-unit table ($V/c16b/units.parquet)
    python c16b_type_matched.py                             # -> $V/c16b/{design1,design2}_{summary,differences}.tsv, verdict.json
Design 1: cap 5 calls per cell type per bin; per comparison and bin, types where both methods have >= 5 scorable units; each
method's own top 5 per type. Design 2: check 16 primary call sets restricted to types where every compared method has a call."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402

V16, OUT = os.path.join(vlib.V, "c16"), os.path.join(vlib.V, "c16b"); CAP = 5
BINS = ["1-19", "20-99", "100-499", ">=500"]; COMPARED = ["20-99", "100-499", ">=500"]
LZ, RAW, LC = "LeafletFA_z", "raw_pseudobulk", "LeafCutter"


def boot_excess(calls_by_method, types):
    """type bootstrap (2,000, seed 0) over `types`: each method's excess on its own calls in the resampled types"""
    types = np.array(sorted(types)); pos = {k: i for i, k in enumerate(types)}
    br = np.random.default_rng(0).integers(0, len(types), (2000, len(types))); w = np.stack([np.bincount(b, minlength=len(types)) for b in br])
    out = {}
    for m, t in calls_by_method.items():
        i = t.k.map(pos).values; sm = np.bincount(i, t.excess.values, len(types)); ct = np.bincount(i, minlength=len(types)).astype(float)
        out[m] = (w @ sm) / np.maximum(w @ ct, 1e-12)
    return out


def score(design, sets):
    """sets: list of (bin, comparator, types, {method: calls}) -> summary rows and difference rows"""
    S, D = [], []
    for b, comp, types, cm in sets:
        bx = boot_excess(cm, types) if len(types) else {}
        for m, t in cm.items():
            S.append(dict(design=design, bin=b, comparison=f"{LZ} vs {comp}", method=m, types=len(types), calls=len(t),
                          real=t.real.mean(), null=t.null.mean(), excess=t.excess.mean(),
                          excess_lo=np.quantile(bx[m], .025) if len(types) else np.nan, excess_hi=np.quantile(bx[m], .975) if len(types) else np.nan))
        if len(types):
            dd = bx[LZ] - bx[comp]
            D.append(dict(design=design, bin=b, comparator=comp, types=len(types), diff=cm[LZ].excess.mean() - cm[comp].excess.mean(),
                          diff_lo=np.quantile(dd, .025), diff_hi=np.quantile(dd, .975)))
        else:
            D.append(dict(design=design, bin=b, comparator=comp, types=0, diff=np.nan, diff_lo=np.nan, diff_hi=np.nan))
    return pd.DataFrame(S), pd.DataFrame(D)


def readings(Df):
    P = Df[Df.bin.isin(COMPARED) & Df.types.gt(0)]
    noninf = bool(len(P) and (P.diff_lo > -0.05).all())
    better = noninf and all((P[P.comparator == c].diff_lo > 0).any() for c in (RAW, LC) if (P.comparator == c).any())
    g = lambda b, c: Df[(Df.bin == b) & (Df.comparator == c)]
    thin_raw = {b: bool(len(g(b, RAW)) and g(b, RAW).diff_lo.iloc[0] > 0) for b in ("1-19", "20-99")}
    lc = {b: bool(len(g(b, LC)) and g(b, LC).types.iloc[0] > 0 and g(b, LC).diff_lo.iloc[0] > 0) for b in ("100-499", ">=500")}
    return dict(non_inferior=noninf, better=better, verdict="BETTER" if better else ("NON-INFERIOR" if noninf else "FAIL"),
                raw_lower_bound_above_0=thin_raw, leafcutter_lower_bound_above_0=lc, compared=[f"{r.bin}:{r.comparator}" for r in P.itertuples()])


def main():
    a, b = pd.read_csv(os.path.join(V16, "summary.tsv"), sep="\t"), pd.read_csv(os.path.join(OUT, "summary.tsv"), sep="\t")
    pd.testing.assert_frame_equal(a, b)                                          # the regenerated check 16 equals the reported one
    A = pd.read_parquet(os.path.join(OUT, "units.parquet")); A["excess"] = A.real - A.null
    C16 = pd.read_csv(os.path.join(OUT, "calls.tsv.gz"), sep="\t"); C16 = C16[C16.design == "primary"][["k", "a", "method"]]
    C16 = C16.merge(A, on=["k", "a", "method"], how="left", validate="1:1"); assert C16.real.notna().all()   # exact values from the unit table
    # ---- design 1: cap 5 per type, same types for both methods of a comparison ----
    sets1 = []
    for bn in BINS:
        sub = A[(A.bin == bn) & A.scorable]
        for comp in (RAW, LC):
            if bn == "1-19" and comp == LC:
                continue
            n = sub[sub.method.isin([LZ, comp])].groupby(["method", "k"]).size().unstack(0).reindex(columns=[LZ, comp]).fillna(0)
            types = n.index[(n >= CAP).all(1)].values
            cm = {}
            for m in (LZ, comp):
                t = sub[(sub.method == m) & sub.k.isin(types)].copy()
                t = t.iloc[np.lexsort((np.arange(len(t)), -t.tie.values, -t.stat.values))]   # check 16 order: stat desc, tie desc, (k, a)
                cm[m] = t.groupby("k", sort=False).head(CAP)
            sets1.append((bn, comp, types, cm))
    S1, D1 = score("design1_capped", sets1)
    # ---- design 2: check 16 primary calls, types where every compared method has >= 1 call ----
    sets2 = []
    for bn in BINS:
        ms = [LZ, RAW] + ([LC] if bn in ("100-499", ">=500") else [])
        cb = C16[(C16.bin == bn) & C16.method.isin(ms)]
        types = sorted(set.intersection(*[set(cb[cb.method == m].k) for m in ms]))
        cm = {m: cb[(cb.method == m) & cb.k.isin(types)] for m in ms}
        for comp in ms[1:]:
            sets2.append((bn, comp, types, {LZ: cm[LZ], comp: cm[comp]}))
    S2, D2 = score("design2_restricted", sets2)
    for nm, S, D in (("design1", S1, D1), ("design2", S2, D2)):
        S.to_csv(os.path.join(OUT, f"{nm}_summary.tsv"), sep="\t", index=False, float_format="%.4f")
        D.to_csv(os.path.join(OUT, f"{nm}_differences.tsv"), sep="\t", index=False, float_format="%.4f")
    r1, r2 = readings(D1), readings(D2)
    conc = "not explained by concentration" if all(r["raw_lower_bound_above_0"][b] for r in (r1, r2) for b in ("1-19", "20-99")) \
        else "at least partly explained by concentration"
    lcr = all(r["leafcutter_lower_bound_above_0"][b] for r in (r1, r2) for b in ("100-499", ">=500"))
    V = dict(cap=CAP, design1=r1, design2=r2, thin_advantage_over_raw=conc, leafcutter_advantage_holds_in_both=lcr)
    json.dump(V, open(os.path.join(OUT, "verdict.json"), "w"), indent=1)
    print(S1.round(3).to_string(index=False)); print(D1.round(3).to_string(index=False))
    print(S2.round(3).to_string(index=False)); print(D2.round(3).to_string(index=False)); print(json.dumps(V, indent=1))


if __name__ == "__main__":
    main()
