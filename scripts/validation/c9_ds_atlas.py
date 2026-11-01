"""Check 9a' (CPU, read-only on the shipped outputs): reproducibility of GB Fig. 3c's posterior DS calls across the 3
pooled-atlas K20 seeds (pre-registration amendment 5a).

    python c9_ds_atlas.py

Call sets are the shipped ones: per-seed A51 outputs (delta 0.15, Bayesian FDR over all ATSEs), q <= 0.05, then the A67
gate (>= 25 covered cells per arm). Contrasts: the 5 with the most gated consensus calls in GB Fig. 3c.
Reported: pairwise Jaccard per contrast (3 pairs), fraction of seed-0 calls called in both other seeds, Pearson r of
per-cell rho = phi . psi between seeds over all cells of the two arms and the junctions of the called events.
Pass: median pairwise Jaccard >= 0.7. Writes tsv/c9a2_contrasts.tsv, c9a2_pairs.tsv.
"""
import itertools, os, re, sys
import numpy as np, pandas as pd, h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa")
from _phi_rows import phi_row_of  # noqa: E402

M = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
CMP = f"{M}/LEAFCUTTER_CMP_DELTAONLY"
A51 = {0: f"{CMP}/a51_delta0.15/a51_group_ds_posterior.tsv", 1: f"{CMP}/a51_seed1_delta0.15/a51_group_ds_posterior.tsv",
       2: f"{CMP}/a51_seed2_delta0.15/a51_group_ds_posterior.tsv"}
FITS = {s: f"{M}/ATLAS_DM/runs_final_deltaonly/dm_simplex_K20_seed{s}_b8192/fit.npz" for s in A51}
H5 = f"{M}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"
GATED = f"{CMP}/analysis_matched/posterior_consensus_gated.tsv.gz"
T = os.path.join(vlib.V, "tsv")


def norm(s):
    return re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")


def main():
    G = pd.read_csv(GATED, sep="\t", usecols=["contrast", "cluster", "lfa_cons", "eligible"])
    top = (G[G.eligible].groupby("contrast").lfa_cons.sum().sort_values(ascending=False).head(5))
    print(top.to_string())
    elig = G[G.eligible & G.contrast.isin(top.index)][["contrast", "cluster"]]
    calls = {}
    for s, p in A51.items():
        d = pd.read_csv(p, sep="\t", usecols=["contrast", "cluster", "q"])
        d = d[d.contrast.isin(top.index)].merge(elig, on=["contrast", "cluster"])
        calls[s] = {c: set(g.cluster[g.q <= 0.05]) for c, g in d.groupby("contrast")}
    with h5py.File(H5, "r") as f:
        g = f["obs"]["tissue_cell_type"]
        cats = np.array([x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]])
        lab = np.array([norm(x) for x in cats])[g["codes"][:]]
    Z = {s: np.load(p, allow_pickle=True) for s, p in FITS.items()}
    ev = np.asarray(Z[0]["event_id"]).astype(str)[np.asarray(Z[0]["inv_ev"])]
    for s in (1, 2):
        assert np.array_equal(np.asarray(Z[s]["event_id"]).astype(str)[np.asarray(Z[s]["inv_ev"])], ev)
    prow = {s: phi_row_of(Z[s], len(lab)) for s in Z}
    rows, crow = [], []
    for c in top.index:
        a, b = c.split("__vs__")
        cells = np.flatnonzero(np.isin(lab, [norm(a), norm(b)]))
        cells = cells[np.all([prow[s][cells] >= 0 for s in Z], axis=0)]
        called = set().union(*[calls[s].get(c, set()) for s in Z])
        cols = np.flatnonzero(np.isin(ev, list(called)))
        rho = {s: (np.asarray(Z[s]["phi"])[prow[s][cells]].astype(np.float64)
                   @ np.asarray(Z[s]["psi_mean"])[:, cols].astype(np.float64)).ravel() for s in Z}
        for x, y in itertools.combinations(sorted(Z), 2):
            A, B = calls[x].get(c, set()), calls[y].get(c, set())
            rows.append(dict(contrast=c, seed_a=x, seed_b=y, n_a=len(A), n_b=len(B), n_both=len(A & B),
                             jaccard=len(A & B) / len(A | B) if A | B else np.nan, rho_r=np.corrcoef(rho[x], rho[y])[0, 1]))
        s0 = calls[0].get(c, set())
        crow.append(dict(contrast=c, gb_gated_consensus=int(top[c]), calls_seed0=len(s0),
                         calls_seed1=len(calls[1].get(c, set())), calls_seed2=len(calls[2].get(c, set())),
                         frac_seed0_in_both_others=np.mean([x in calls[1].get(c, set()) and x in calls[2].get(c, set()) for x in s0]) if s0 else np.nan,
                         n_cells=len(cells), n_junctions=len(cols)))
    P = pd.DataFrame(rows); C = pd.DataFrame(crow).merge(
        P.groupby("contrast").agg(jaccard_median=("jaccard", "median"), rho_r_median=("rho_r", "median")).reset_index())
    P.to_csv(os.path.join(T, "c9a2_pairs.tsv"), sep="\t", index=False)
    C.to_csv(os.path.join(T, "c9a2_contrasts.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(C.round(3).to_string(index=False)); print(P.round(3).to_string(index=False))
    med = float(P.jaccard.median())
    print(f"\nmedian pairwise Jaccard over {len(P)} pairs: {med:.3f} -> {'PASS' if med >= 0.7 else 'FAIL'} (>= 0.7)")


if __name__ == "__main__":
    main()
