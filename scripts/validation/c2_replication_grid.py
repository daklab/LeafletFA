"""Check 2 extension (CPU): program replication across seeds, at several K, on existing fits.

    python c2_replication_grid.py --grid <dir with dm_simplex_K{K}_seed{s}_b8192/fit.npz> --Ks 10,20,30 --tag TMS70

The A70 rule with liveness (prereg amendment 2): psi centred by its pi-weighted mean composition,
Hungarian on cosine with seed 0 as reference; a program replicates if live (argmax of >= 1 % of the
fit's cells) in all three seeds and matched at centred cosine >= 0.8 in both other seeds.
Writes tsv/c2_replication_<tag>.tsv (per program) and tsv/c2_replication_<tag>_summary.tsv.
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402


def load(path):
    z = np.load(path, allow_pickle=True)
    psi = z["psi_mean"].astype(np.float64)
    pi = np.asarray(z["pi"] if "pi" in z.files else z["trace_pi"][-1], np.float64); pi = pi / pi.sum()
    phi = z["phi"]
    occ = np.bincount(np.asarray(phi).argmax(1), minlength=psi.shape[0]) / len(phi)
    C = psi - pi @ psi
    return C / np.linalg.norm(C, axis=1, keepdims=True).clip(1e-12), occ, pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--Ks", default="10,20,30")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--pattern", default="dm_simplex_K{K}_seed{s}_b8192/fit.npz")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--live", type=float, default=0.01)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    rows, summ = [], []
    for K in [int(k) for k in a.Ks.split(",")]:
        fits = {s: load(os.path.join(a.grid, a.pattern.format(K=K, s=s))) for s in seeds}
        Cr, occ_r, pi_r = fits[seeds[0]]
        res = pd.DataFrame({"K": K, "program_seed0": np.arange(K) + 1, "pi_seed0": pi_r,
                            "live_seed0": occ_r >= a.live})
        for s in seeds[1:]:
            Cs, occ_s, _ = fits[s]
            R = Cr @ Cs.T
            i, j = linear_sum_assignment(-R)
            cos = np.full(K, np.nan); cos[i] = R[i, j]
            lv = np.zeros(K, bool); lv[i] = occ_s[j] >= a.live
            res[f"cos_seed{s}"] = cos; res[f"live_seed{s}"] = lv
        lv_all = res[[c for c in res if c.startswith("live_seed")]].all(1)
        res["min_cos"] = res[[c for c in res if c.startswith("cos_seed")]].min(1)
        res["replicates"] = lv_all & (res["min_cos"] >= 0.8)
        res["replicates_ignoring_liveness"] = res["min_cos"] >= 0.8
        rows.append(res)
        summ.append(dict(tag=a.tag, K=K, live_seed0=int(res["live_seed0"].sum()), live_all_seeds=int(lv_all.sum()),
                         replicating=int(res["replicates"].sum()),
                         replicating_ignoring_liveness=int(res["replicates_ignoring_liveness"].sum()),
                         median_min_cos_live=float(res.loc[lv_all, "min_cos"].median()) if lv_all.any() else np.nan))
    out = os.path.join(vlib.V, "tsv")
    pd.concat(rows).to_csv(os.path.join(out, f"c2_replication_{a.tag}.tsv"), sep="\t", index=False)
    s = pd.DataFrame(summ); s.to_csv(os.path.join(out, f"c2_replication_{a.tag}_summary.tsv"), sep="\t", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
