"""Read-only (Karin 2026-11-06), browser gap part 2: gap between LeafletFA and raw delta psi by rank depth, against TRAIN and held-out TEST
pseudobulk, for the two browser example pairs and 20 random eligible pairs per atlas (seed 7).
    python c10_browser_gap_rank.py <TMS|Allen> "<type A>" "<type B>"
(a) Same table: the browser loads c10/<ds>_s0_estimates.npz `point`, saved at c10_catalog.py:342 from the `point` that 10.1 scores
    (c10_catalog.py:291); asserted here by recomputing point from the fit.
(b) Per pair, ATSEs passing the browser filter (>= 20 TRAIN reads and >= 3 TRAIN cells with reads in both types), ranked by LeafletFA max
    |dpsi|; lead junction = argmax. Rank bins: top 100, 101-1,000, all. Per bin: median |dL - dTR| (the browser gap), and on rows with
    >= 20 TEST reads and >= 3 TEST cells in both types: median |dL - dTE|, |dTR - dTE|, sign(dL) == sign(dTE), and median |dL| - |dTE|
    (shrink toward TEST = regression of extremes). Same table ranked by raw TRAIN |dpsi| for comparison.
(c) Reading, fixed before results: bug if any assert fails. Otherwise, at the top 100 (TEST-evaluable rows):
    - selection: |dL| - |dTE| > 0 at the top and near 0 for "all", and LeafletFA's error vs TEST at the top is not larger than raw TRAIN's
      error vs TEST when raw ranks its own top 100;
    - smoothing: LeafletFA's error vs TEST at the top exceeds raw TRAIN's error vs TEST on the same rows;
    - raw noise: raw TRAIN's error vs TEST on the same rows exceeds LeafletFA's, with the gap concentrated in low-read rows.
    More than one can hold; all are reported."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
from c10_catalog import Data, Groups, type_setup, estimates, raw_pb, log  # noqa: E402
ds, ta, tb = sys.argv[1:4]; OUT = os.path.join(vlib.V, "c10_browser_gap"); os.makedirs(OUT, exist_ok=True)
D = Data(ds, 0); types, g = type_setup(D); G = len(types) + 1
E = np.load(os.path.join(vlib.V, "c10", f"{ds}_s0_estimates.npz"), allow_pickle=True); assert list(E["types"].astype(str)) == list(types)
p, *_ = estimates(D, Groups(D, np.arange(len(g)), g, G), draws=False); bug_a = float(np.abs(p[:-1] - E["point"]).max()); assert bug_a < 1e-4, bug_a
P = E["point"]; y, n = raw_pb(D.Y, D.Ncell, D.inv, np.arange(len(g)), g, G)
with np.errstate(invalid="ignore", divide="ignore"): TR = y[:-1] / n[:-1]
TRe = E["reads"]; TRc = E["cov"]
D.load_test(ds); te = D.test; code = {t: i for i, t in enumerate(types)}; gt = np.array([code.get(t, G - 1) for t in te["t"]])
O = sp_onehot = None
import scipy.sparse as sp
Ot = sp.csr_matrix((np.ones(len(gt)), (gt, np.arange(len(gt)))), shape=(G, len(gt)))
yt = np.asarray((Ot @ te["Y"]).todense())[:-1]; nt = np.asarray((Ot @ te["N"]).todense())[:-1]
ct = np.asarray((Ot @ (te["N"] > 0).astype(np.float64)).todense())[:-1]
with np.errstate(invalid="ignore", divide="ignore"): TE = yt / nt[:, D.inv]
rng = np.random.default_rng(7); elig = np.flatnonzero(E["eligible"]); pairs = [(code[ta], code[tb])]
while len(pairs) < 21:
    a, b = rng.choice(elig, 2, replace=False)
    if (a, b) not in pairs: pairs.append((int(a), int(b)))
rows = []
def evaluate(a, b, rank_by, tag):
    ok = (TRe[a] >= 20) & (TRc[a] >= 3) & (TRe[b] >= 20) & (TRc[b] >= 3)
    dL = P[a] - P[b]; dTR = TR[a] - TR[b]; dTE = TE[a] - TE[b]; key = np.abs(dL if rank_by == "LeafletFA" else dTR)
    f = pd.DataFrame(dict(e=D.inv, k=key, j=np.arange(D.J)))[ok[D.inv]].sort_values("k", ascending=False).drop_duplicates("e")
    f["rank"] = np.arange(1, len(f) + 1); j = f.j.values; e = f.e.values
    f["dL"], f["dTR"], f["dTE"] = dL[j], dTR[j], dTE[j]
    f["te_ok"] = (nt[a, e] >= 20) & (ct[a, e] >= 3) & (nt[b, e] >= 20) & (ct[b, e] >= 3); f["tr_reads_min"] = np.minimum(TRe[a, e], TRe[b, e])
    for lab, m in (("top100", f["rank"] <= 100), ("101-1000", (f["rank"] > 100) & (f["rank"] <= 1000)), ("all", f["rank"] > 0)):
        x = f[m]; z = x[x.te_ok]
        rows.append(dict(dataset=ds, pair=f"{types[a]} vs {types[b]}", example=(a, b) == pairs[0], ranked_by=rank_by, bin=lab, n=len(x), n_test=len(z),
                         gap_L_vs_TR=float((x.dL - x.dTR).abs().median()), err_L_vs_TE=float((z.dL - z.dTE).abs().median()), err_TR_vs_TE=float((z.dTR - z.dTE).abs().median()),
                         sign_L_TE=float((np.sign(z.dL) == np.sign(z.dTE)).mean()), sign_TR_TE=float((np.sign(z.dTR) == np.sign(z.dTE)).mean()),
                         shrink_L=float((z.dL.abs() - z.dTE.abs()).median()), shrink_TR=float((z.dTR.abs() - z.dTE.abs()).median()),
                         gap_low_reads=float((x.dL - x.dTR).abs()[x.tr_reads_min < 100].median()), gap_high_reads=float((x.dL - x.dTR).abs()[x.tr_reads_min >= 100].median())))
    if (a, b) == pairs[0] and rank_by == "LeafletFA": f.head(1000).to_csv(os.path.join(OUT, f"{ds}_rank_example_top1000.tsv"), sep="\t", index=False, float_format="%.4g")
for a, b in pairs:
    for rb in ("LeafletFA", "raw TRAIN"): evaluate(a, b, rb, "")
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, f"{ds}_rank_table.tsv"), sep="\t", index=False, float_format="%.4f")
num = ["n", "n_test", "gap_L_vs_TR", "err_L_vs_TE", "err_TR_vs_TE", "sign_L_TE", "sign_TR_TE", "shrink_L", "shrink_TR", "gap_low_reads", "gap_high_reads"]
print(f"(a) recomputed point = saved point (max abs diff {bug_a:.2g}); browser and 10.1 read the same array")
print("example pair:"); print(R[R.example][["ranked_by", "bin"] + num].round(3).to_string(index=False))
print("20 random eligible pairs, median over pairs:"); print(R[~R.example].groupby(["ranked_by", "bin"], sort=False)[num].median().round(3).to_string())
