"""Check 2 summary (CPU): per K / prior, kNN (real, content-only, exact average, drop), live programs,
participation ratio of the embedding, event-split gain over null (i); then the amendment-1 rules.

    python c2_summary.py

Inputs: check2/TMS_<cfg>/knn_check.tsv and tsv/c1_TMS_<cfg>_gains.tsv (run knn_check.py and
c1_score.py per config first); K = 20 = the baseline seed-0 run and tsv/c1_TMS_K20_s0_gains.tsv.
Writes tsv/c2_K_prior.tsv and prints the rule outcomes.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402

CFGS = [("K5", 5), ("K10", 10), ("K20", 20), ("K30", 30), ("K20_priorfloor1", 20)]


def pr(phi):
    cl = np.log(np.clip(phi, 1e-6, None)); cl -= cl.mean(1, keepdims=True)
    lam = np.clip(np.linalg.eigvalsh(np.cov(cl.T)), 0, None)
    return lam.sum() ** 2 / (lam ** 2).sum()


def main():
    rows = []
    for cfg, K in CFGS:
        run = os.path.join(vlib.B, "TMS", "full_K20_s0_f64") if cfg == "K20" else os.path.join(vlib.V, "check2", f"TMS_{cfg}")
        gtag = "K20_s0" if cfg == "K20" else cfg
        kp, gp = os.path.join(run, "knn_check.tsv"), os.path.join(vlib.V, "tsv", f"c1_TMS_{gtag}_gains.tsv")
        if not (os.path.exists(kp) and os.path.exists(gp)):
            print(f"{cfg}: not ready"); continue
        k = pd.read_csv(kp, sep="\t").set_index("input")
        g = pd.read_csv(gp, sep="\t").set_index("comparison")
        phi = np.load(os.path.join(run, "factors_real.npz"))["phi"].astype(np.float64)
        rows.append(dict(config=cfg, K=K, knn_real=k.loc["real", "knn_acc"], knn_contentonly=k.loc["contentonly", "knn_acc"],
                         knn_exact_average=k.loc["popavg", "knn_acc"],
                         drop=k.loc["real", "knn_acc"] - k.loc["contentonly", "knn_acc"],
                         live_programs=int(k.loc["real", "live_programs"]), participation_ratio=pr(phi),
                         gain_vs_null_i=g.loc["model_split - null_i_population", "gain_per_read"],
                         gain_ci=f'[{g.loc["model_split - null_i_population", "ci_lo"]:.5f}, {g.loc["model_split - null_i_population", "ci_hi"]:.5f}]'))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(vlib.V, "tsv", "c2_K_prior.tsv"), sep="\t", index=False)
    print(t.round(4).to_string(index=False))
    if {"K10", "K20", "K30"} <= set(t.config):
        r = t.set_index("config")
        sim = all(abs(r.loc[c, "knn_real"] - r.loc["K20", "knn_real"]) <= 0.03 and abs(r.loc[c, "drop"] - r.loc["K20", "drop"]) <= 0.03
                  for c in ("K10", "K30"))
        seed_range = float(np.ptp(pd.read_csv(os.path.join(vlib.V, "tsv", "c3_TMS_seed_knn.tsv"), sep="\t")["real"].values))
        better30 = r.loc["K30", "knn_real"] - r.loc["K20", "knn_real"] > max(0.03, seed_range)
        unused = (r.loc["K30", "live_programs"] - r.loc["K20", "live_programs"] <= 3 and
                  r.loc["K30", "participation_ratio"] < 1.2 * r.loc["K20", "participation_ratio"] and
                  r.loc["K30", "knn_real"] - r.loc["K20", "knn_real"] <= 0.03)
        best_gain = r.loc[["K5", "K10", "K20", "K30"], "gain_vs_null_i"].max() if "K5" in r.index else r.loc[["K10", "K20", "K30"], "gain_vs_null_i"].max()
        best_knn = r.loc[[c for c in ("K5", "K10", "K20", "K30") if c in r.index], "knn_real"].max()
        k20_ok = (r.loc["K20", "gain_vs_null_i"] >= best_gain * 0.99) or (r.loc["K20", "knn_real"] >= best_knn - 0.02)
        print(f"\nrule 'similar at K = 10 and 30': {sim}\nrule 'K30 replaces K20 in Fig 1': {better30} (seed range {seed_range:.3f})"
              f"\nrule 'higher K adds unused programs, not information': {unused}\ncheck 2 pass (K20 within 1% of best gain or 0.02 of best kNN): {k20_ok}")


if __name__ == "__main__":
    main()
