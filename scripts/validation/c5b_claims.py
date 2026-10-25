"""Check 5b (CPU): do the manuscript's key numbers reproduce from the current code and the saved
fits/outputs, with no retraining?

    python c5b_claims.py

Two kinds of check, labelled in the output:
  recomputed  -- the number is recomputed from saved fits with current LeafletFA code
  from_output -- the number is re-read from the saved analysis output and compared with the text
Figure build scripts are NOT rerun: they write into the Genome Biology figure folders.
Writes tsv/c5b_claims.tsv.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402

W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD"
M = f"{W}/MOUSE_SPLICING_FOUNDATION"
FINAL = f"{M}/ATLAS_DM/runs_final_deltaonly"


def raw_replication(grid, K, seeds=(0, 1, 2), thr=0.9):
    """A61's definition as stated in Methods L131: Hungarian on |corr| of psi, >= thr in every other seed."""
    P = {s: np.load(f"{grid}/dm_simplex_K{K}_seed{s}_b8192/fit.npz", allow_pickle=True)["psi_mean"].astype(np.float64) for s in seeds}
    Z = {s: (p - p.mean(1, keepdims=True)) / p.std(1, keepdims=True) for s, p in P.items()}
    ok = np.ones(K, bool)
    for s in seeds[1:]:
        R = np.abs(Z[seeds[0]] @ Z[s].T / Z[s].shape[1])
        i, j = linear_sum_assignment(-R)
        r = np.zeros(K); r[i] = R[i, j]
        ok &= r >= thr
    return int(ok.sum())


def main():
    rows = []

    def add(fig, claim, stated, found, kind, ok, note=""):
        rows.append(dict(figure=fig, claim=claim, stated=stated, found=found, kind=kind,
                         status="reproduced" if ok else "NOT reproduced", note=note))

    r = json.load(open(f"{FINAL}/dm_simplex_K20_seed0_b8192/result.json"))
    dims = f"{r['n_cells']:,} x {r['n_junctions']:,} x {r['n_atse']:,}"
    add("1", "atlas size (cells x junctions x ATSEs)", "132,706 x 79,768 x 24,475", dims, "from_output",
        dims == "132,706 x 79,768 x 24,475")

    z = np.load(f"{FINAL}/dm_simplex_K20_seed0_b8192/fit.npz", allow_pickle=True)
    c = z["bb_conc_vec"]
    add("Methods", "share of per-ATSE concentrations at the floor of 2", "75%", f"{(c <= 2.0001).mean():.1%}",
        "recomputed", abs((c <= 2.0001).mean() - 0.75) < 0.01)

    for K, stated in ((20, "11-14/20 (atlas 11)"),):
        n = raw_replication(FINAL, K)
        add("4 / S5", f"programs replicating across seeds, atlas K{K} (raw psi |r| >= 0.9, Methods L131)", stated,
            f"{n}/{K}", "recomputed", n == 11,
            "metric is inflated by shared usage (A70); centred-psi cosine gives 4/20 (A70) or 1/20 with argmax liveness (check 2)")

    a50 = pd.read_csv(f"{M}/LEAFCUTTER_CMP_DELTAONLY/analysis_group/a50_contrast_comparison.tsv", sep="\t")
    found = f"mean {a50.lfa_cal.mean():.0f} vs {a50.lc_cal.mean():.0f}; median {a50.lfa_cal.median():.0f} vs {a50.lc_cal.median():.0f}"
    add("3b", "calls per contrast at matched realised FDP, LeafletFA vs LeafCutter", "840 vs 641", found, "from_output",
        a50.lfa_cal.median() == 840 and int(a50.lc_cal.median()) == 641,
        f"median over {len(a50)} contrasts (LeafCutter 641.5); text should say 'median'")

    g = pd.read_csv(f"{M}/LEAFCUTTER_CMP_DELTAONLY/analysis_matched/posterior_consensus_gated.tsv.gz", sep="\t",
                    usecols=["votes", "eligible", "lc_sig", "lfa_cons"])
    ung = int(g.lfa_cons.astype(bool).sum()); gated = int((g.lfa_cons.astype(bool) & g.eligible.astype(bool)).sum())
    shared = int((g.lfa_cons.astype(bool) & g.eligible.astype(bool) & (g.lc_sig.astype(str) == "True")).sum())
    add("3c", "posterior calls ungated -> coverage-gated; shared with LeafCutter", "49,865 -> 7,041; 3,551 shared",
        f"{ung:,} -> {gated:,}; {shared:,} shared", "from_output", ung == 49865 and gated == 7041 and shared == 3551)

    a69 = pd.read_csv(f"{M}/ATLAS_DM/aging/a69_dm_simplex_K20_seed0_b8192_program_age.tsv", sep="\t")
    sp4 = a69[a69.SP == "SP4"].iloc[0]
    add("4b", "aging program: per-mouse p (males 3m vs 24m); share of types up; mean 3m -> 24m",
        "P = 0.0159; 94%; 0.13 -> 0.22",
        f"P = {sp4.p_mouse_3v24_male:.4f}; {sp4.frac_types_up:.0%}; {sp4.mean_3m:.2f} -> {sp4.mean_24m:.2f}", "from_output",
        abs(sp4.p_mouse_3v24_male - 0.0159) < 1e-3 and round(sp4.frac_types_up, 2) >= 0.93 and round(sp4.mean_24m, 2) == 0.22)

    pm = pd.read_csv(f"{W}/CROSS_SPECIES_2026/program_match_mouse_human.tsv", sep="\t")
    best = pm.sort_values("r", ascending=False).groupby(["mouse_seed", "human_seed"]).head(1)
    add("5", "conserved pair (best mouse-human match in each of the 9 seed pairs), r", "0.586-0.589",
        f"{best.r.min():.3f}-{best.r.max():.3f} over {len(best)} seed pairs", "from_output",
        abs(best.r.min() - 0.586) < 0.0015 and abs(best.r.max() - 0.589) < 0.0015,
        "programs are numbered per seed (SP13/SP14 mouse, SP8/SP9 human); stated range too narrow: 0.581-0.589")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(vlib.V, "tsv", "c5b_claims.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(out[["figure", "claim", "stated", "found", "kind", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
