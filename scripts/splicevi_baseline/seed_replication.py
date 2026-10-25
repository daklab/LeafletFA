"""Across-seed summary of the SpliceVI-baseline fits: kNN per seed and program replication.

    python seed_replication.py --root <runs>/<dataset> --seeds 0,1,2

kNN: reads knn_check.tsv of each full_K20_s<seed>_f64 run; reports real / content-only / exact average
per seed with mean and range.

Replication: for each pair of seeds, programs are matched one-to-one by the Hungarian algorithm on the
Pearson correlation of their psi rows (K x junctions, within-ATSE compositions). Reported for all K
programs and for LIVE programs only (argmax of >= 1 % of the training cells in BOTH seeds of the pair):
a dead program stays near its prior, and dead-to-dead matches would inflate agreement. Writes
seed_knn.tsv and seed_replication.tsv under --root.
"""
import argparse, itertools, os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--live", type=float, default=0.01)
    a = ap.parse_args()
    from scipy.optimize import linear_sum_assignment
    seeds = [int(s) for s in a.seeds.split(",")]
    run = lambda s: os.path.join(a.root, f"full_K20_s{s}_f64")

    k = []
    for s in seeds:
        t = pd.read_csv(os.path.join(run(s), "knn_check.tsv"), sep="\t").set_index("input")
        k.append(dict(seed=s, real=t.loc["real", "knn_acc"], contentonly=t.loc["contentonly", "knn_acc"],
                      exact_average=t.loc["popavg", "knn_acc"],
                      drop_contentonly=t.loc["real", "knn_acc"] - t.loc["contentonly", "knn_acc"]))
    k = pd.DataFrame(k)
    summ = k.drop(columns="seed").agg(["mean", "min", "max"]).T.round(3)
    k.to_csv(os.path.join(a.root, "seed_knn.tsv"), sep="\t", index=False)
    print(k.round(3).to_string(index=False)); print(summ.to_string())

    fits = {s: np.load(os.path.join(run(s), "fit", "fit.npz"), allow_pickle=True) for s in seeds}

    # PRIMARY (prereg amendment 2, = Leaflet-analysis A70): psi centred by its pi-weighted mean
    # composition, Hungarian on cosine; a seed-0 program replicates if live and >= 0.8 in BOTH others.
    occ = lambda z: np.bincount(z["phi"].argmax(1), minlength=z["psi_mean"].shape[0]) / len(z["phi"])

    def centred(z):
        P = z["psi_mean"].astype(np.float64); pi = np.asarray(z["pi"], np.float64); pi = pi / pi.sum()
        C = P - pi @ P
        return C / np.linalg.norm(C, axis=1, keepdims=True).clip(1e-12)
    ref = seeds[0]; Cr = centred(fits[ref]); live_r = occ(fits[ref]) >= a.live
    best = {}
    for s in seeds[1:]:
        Cs = centred(fits[s]); R = Cr @ Cs.T
        i, j = linear_sum_assignment(-R)
        cos = np.full(Cr.shape[0], np.nan); cos[i] = R[i, j]
        live_s = np.zeros(Cr.shape[0], bool); live_s[i] = occ(fits[s])[j] >= a.live
        best[s] = (cos, live_s)
    cen = pd.DataFrame({"program_seed%d" % ref: np.arange(Cr.shape[0]) + 1,
                        "live_seed%d" % ref: live_r, "pi_seed%d" % ref: fits[ref]["pi"]})
    for s, (cos, ls) in best.items():
        cen[f"centred_cos_seed{s}"] = cos; cen[f"live_seed{s}"] = ls
    cc = cen[[c for c in cen if c.startswith("centred_cos")]]
    lv = cen[[c for c in cen if c.startswith("live_seed")]].all(1)
    cen["min_centred_cos"] = cc.min(1); cen["replicates"] = lv & (cen["min_centred_cos"] >= 0.8)
    cen.to_csv(os.path.join(a.root, "seed_replication_centred.tsv"), sep="\t", index=False)
    print(f"centred rule: {int(cen['replicates'].sum())} programs replicate "
          f"(live in all seeds: {int(lv.sum())}); median min centred cos over live: "
          f"{cen.loc[lv, 'min_centred_cos'].median():.3f}")
    print(cen.round(3).to_string(index=False))

    # SECONDARY: raw-psi Pearson r per seed pair (inflated by shared usage; reported for comparison)
    rows = []
    for s1, s2 in itertools.combinations(seeds, 2):
        P1, P2 = fits[s1]["psi_mean"].astype(np.float64), fits[s2]["psi_mean"].astype(np.float64)
        Z1 = (P1 - P1.mean(1, keepdims=True)) / P1.std(1, keepdims=True)
        Z2 = (P2 - P2.mean(1, keepdims=True)) / P2.std(1, keepdims=True)
        R = Z1 @ Z2.T / P1.shape[1]
        i, j = linear_sum_assignment(-R)
        r = R[i, j]
        live = (occ(fits[s1])[i] >= a.live) & (occ(fits[s2])[j] >= a.live)
        rows.append(dict(seed_a=s1, seed_b=s2, K=len(r), median_r_all=np.median(r), min_r_all=r.min(),
                         n_live_matched=int(live.sum()),
                         median_r_live=(np.median(r[live]) if live.any() else np.nan),
                         min_r_live=(r[live].min() if live.any() else np.nan),
                         n_live_r_ge_0_8=int((r[live] >= 0.8).sum()),
                         live_r=";".join(f"{x:.2f}" for x in np.sort(r[live])[::-1])))
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(a.root, "seed_replication.tsv"), sep="\t", index=False)
    print(rep.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
