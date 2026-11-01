"""Check 9a (CPU): reproducibility of the shipped posterior DS test across the 15 TMS K20 fits (amendment 5).

    python c9_ds_seeds.py

Contrasts: GB Fig. 3b's 70 TMS contrasts (a50_contrast_comparison.tsv), kept if both arms have >= 50 TMS-only training
cells, ranked by lfa_cal; the contrasts at the min, 25th, 50th, 75th percentile and max (nearest rank).
Per fit and contrast: gated call set. Reported: pairwise Jaccard (105 pairs x 5 contrasts), fraction of seed-0 calls in
>= 12 of 15 fits, cross-seed Pearson r of per-cell rho = phi . psi over the contrast's cells (500 sampled per contrast,
seeded; memory) and the tested ATSEs' junctions. Pass: median Jaccard >= 0.7.
Writes tsv/c9a_contrasts.tsv, c9a_calls.tsv.gz, c9a_jaccard.tsv, c9a_summary.tsv.
"""
import itertools, os, re, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, c9_lib as L  # noqa: E402

A50 = ("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFCUTTER_CMP_DELTAONLY/"
       "analysis_group/a50_contrast_comparison.tsv")
SEEDS, MIN_CELLS, N_RHO, REC = range(15), 50, 500, 12
T = os.path.join(vlib.V, "tsv")


def norm(s):
    return re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")


def pick_contrasts(tct):
    a = pd.read_csv(A50, sep="\t")
    labs = {norm(t): t for t in np.unique(tct)}
    n = pd.Series(tct).value_counts()
    rows = []
    for c, calls in zip(a.contrast, a.lfa_cal):
        g1, g2 = c.split("_VS_")
        t1, t2 = labs.get(g1), labs.get(g2)
        if t1 and t2 and n[t1] >= MIN_CELLS and n[t2] >= MIN_CELLS:
            rows.append(dict(contrast=c, type1=t1, type2=t2, n1=int(n[t1]), n2=int(n[t2]), gb_lfa_cal=int(calls)))
    C = pd.DataFrame(rows).sort_values("gb_lfa_cal").reset_index(drop=True)
    pos = [int(round(q * (len(C) - 1))) for q in (0, 0.25, 0.5, 0.75, 1)]
    out = C.iloc[pos].copy(); out["percentile"] = [0, 25, 50, 75, 100]
    print(f"{len(C)} of {len(a)} GB contrasts eligible in the TMS-only training split")
    return out.reset_index(drop=True)


def main():
    cells, tct, mouse, Y = L.tms_train()
    C = pick_contrasts(tct)
    C.to_csv(os.path.join(T, "c9a_contrasts.tsv"), sep="\t", index=False)
    print(C.to_string(index=False))
    calls, rho = [], {}
    rng = np.random.default_rng(0)
    rho_rows = {c: np.sort(rng.choice(np.flatnonzero(np.isin(tct, [r.type1, r.type2])),
                                      min(N_RHO, int(np.isin(tct, [r.type1, r.type2]).sum())), replace=False))
                for c, r in zip(C.contrast, C.itertuples())}
    post = L.load_posterior(0)
    inv = post["inv"].numpy(); ev = post["event_id"]
    cov = L.coverage(Y, inv, post["n_atse"]).tocsr()
    jcols = {}                                       # the gate is coverage-only, so tested ATSEs do not depend on the fit
    for r in C.itertuples():
        n1 = np.asarray(cov[tct == r.type1].sum(0)).ravel(); n2 = np.asarray(cov[tct == r.type2].sum(0)).ravel()
        jcols[r.contrast] = np.flatnonzero(((n1 >= L.MIN_COV) & (n2 >= L.MIN_COV))[inv])
    for s in SEEDS:
        post = L.load_posterior(s)
        assert np.array_equal(post["cell_id"], cells)
        psi_s = L.draw_psi(post, seed=0)
        for ci, r in enumerate(C.itertuples()):
            r1, r2 = np.flatnonzero(tct == r.type1), np.flatnonzero(tct == r.type2)
            res = L.test(post, psi_s, r1, r2, cov, draw_seed=ci)
            res.insert(0, "seed", s); res.insert(0, "contrast", r.contrast)
            calls.append(res)
            print(f"seed {s} {r.contrast}: {int(res.called.sum())} of {len(res)} gated ATSEs called", flush=True)
            rho[(s, r.contrast)] = (post["phi"][rho_rows[r.contrast]] @ post["psi_mean"][:, jcols[r.contrast]]).astype(np.float32)
        del psi_s
    A = pd.concat(calls, ignore_index=True)
    A.to_csv(os.path.join(T, "c9a_calls.tsv.gz"), sep="\t", index=False)
    jrows, srows = [], []
    for r in C.itertuples():
        sets = {s: set(A[(A.contrast == r.contrast) & (A.seed == s) & A.called].atse) for s in SEEDS}
        for a, b in itertools.combinations(SEEDS, 2):
            u = sets[a] | sets[b]
            ra, rb = rho[(a, r.contrast)].ravel(), rho[(b, r.contrast)].ravel()
            jrows.append(dict(contrast=r.contrast, seed_a=a, seed_b=b, n_a=len(sets[a]), n_b=len(sets[b]),
                              jaccard=len(sets[a] & sets[b]) / len(u) if u else np.nan,
                              rho_r=np.corrcoef(ra, rb)[0, 1]))
        cnt = pd.Series([x for s in SEEDS for x in sets[s]]).value_counts()
        s0 = sets[0]
        srows.append(dict(contrast=r.contrast, gb_lfa_cal=r.gb_lfa_cal, tested_atses_seed0=int(((A.contrast == r.contrast) & (A.seed == 0)).sum()),
                          calls_seed0=len(s0), calls_median=float(np.median([len(sets[s]) for s in SEEDS])),
                          frac_seed0_in_ge12=float(np.mean([cnt.get(x, 0) >= REC for x in s0])) if s0 else np.nan))
    J = pd.DataFrame(jrows); J.to_csv(os.path.join(T, "c9a_jaccard.tsv"), sep="\t", index=False)
    Sm = pd.DataFrame(srows).merge(J.groupby("contrast").agg(jaccard_median=("jaccard", "median"), jaccard_min=("jaccard", "min"),
                                                             jaccard_max=("jaccard", "max"), rho_r_median=("rho_r", "median")).reset_index())
    Sm.to_csv(os.path.join(T, "c9a_summary.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(Sm.round(3).to_string(index=False))
    med = float(J.jaccard.median())
    print(f"\nmedian pairwise Jaccard over {len(J)} fit pairs x contrasts: {med:.3f} -> {'PASS' if med >= 0.7 else 'FAIL'} (>= 0.7)")


if __name__ == "__main__":
    main()
