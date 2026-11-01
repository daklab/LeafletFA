"""Check 9a' POST HOC, descriptive (amendment 5b; the pre-registered 9a' Jaccard result stands).

    python c9_ds_atlas_posthoc.py

Same 5 contrasts and 3 pooled-atlas seeds as c9_ds_atlas.py, shipped A51 outputs, gated ATSEs:
  Spearman between seeds of per-event effect (tv) and posterior probability (prob_de) over all gated ATSEs;
  relaxed replication: called in A (q <= 0.05) -> q < 0.2 in B with the same sign of delta psi at A's lead junction;
  sign agreement at the lead junction for events called in both; the August consensus rule (>= 2 of 3 seeds).
Writes tsv/c9a3_pairs.tsv, c9a3_consensus.tsv.
"""
import itertools, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c9_ds_atlas as A  # noqa: E402


def main():
    G = pd.read_csv(A.GATED, sep="\t", usecols=["contrast", "cluster", "lfa_cons", "eligible"])
    top = G[G.eligible].groupby("contrast").lfa_cons.sum().sort_values(ascending=False).head(5)
    elig = G[G.eligible & G.contrast.isin(top.index)][["contrast", "cluster"]]
    R = {s: pd.read_csv(p, sep="\t", usecols=["contrast", "cluster", "tv", "prob_de", "q"]).merge(elig, on=["contrast", "cluster"])
         for s, p in A.A51.items()}
    import h5py
    with h5py.File(A.H5, "r") as f:
        g = f["obs"]["tissue_cell_type"]
        cats = np.array([x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]])
        lab = np.array([A.norm(x) for x in cats])[g["codes"][:]]
    Z = {s: np.load(p, allow_pickle=True) for s, p in A.FITS.items()}
    ev = np.asarray(Z[0]["event_id"]).astype(str)[np.asarray(Z[0]["inv_ev"])]
    prow = {s: A.phi_row_of(Z[s], len(lab)) for s in Z}
    rows, cons = [], []
    for c in top.index:
        a, b = c.split("__vs__")
        dpsi = {}
        for s in Z:                                                   # signed delta psi per junction, point estimates
            ph = np.asarray(Z[s]["phi"]).astype(np.float64); ps = np.asarray(Z[s]["psi_mean"]).astype(np.float64)
            r1 = np.flatnonzero(lab == A.norm(a)); r2 = np.flatnonzero(lab == A.norm(b))
            r1 = r1[prow[s][r1] >= 0]; r2 = r2[prow[s][r2] >= 0]
            dpsi[s] = (ph[prow[s][r1]].mean(0) - ph[prow[s][r2]].mean(0)) @ ps
        X = {s: R[s][R[s].contrast == c].set_index("cluster") for s in Z}
        ids = X[0].index
        called = {s: set(X[s].index[X[s].q <= 0.05]) for s in Z}

        def lead(s, e):
            j = np.flatnonzero(ev == e); return j[np.argmax(np.abs(dpsi[s][j]))]
        for x, y in itertools.permutations(sorted(Z), 2):
            ca = sorted(called[x])
            rel = [X[y].loc[e, "q"] < 0.2 and np.sign(dpsi[y][lead(x, e)]) == np.sign(dpsi[x][lead(x, e)]) for e in ca]
            both = sorted(called[x] & called[y])
            sg = [np.sign(dpsi[y][lead(x, e)]) == np.sign(dpsi[x][lead(x, e)]) for e in both]
            row = dict(contrast=c, seed_a=x, seed_b=y, n_called_a=len(ca), relaxed_replication=np.mean(rel) if rel else np.nan,
                       n_called_both=len(both), sign_agreement_both=np.mean(sg) if sg else np.nan)
            if x < y:
                row["spearman_tv"] = spearmanr(X[x].loc[ids, "tv"], X[y].loc[ids, "tv"]).correlation
                row["spearman_prob"] = spearmanr(X[x].loc[ids, "prob_de"], X[y].loc[ids, "prob_de"]).correlation
            rows.append(row)
        votes = pd.Series({e: sum(e in called[s] for s in Z) for e in ids})
        cset = set(votes.index[votes >= 2])
        cons.append(dict(contrast=c, n_tested=len(ids), consensus_ge2=len(cset), unanimous=int((votes == 3).sum()),
                         **{f"frac_seed{s}_calls_in_consensus": len(called[s] & cset) / len(called[s]) if called[s] else np.nan for s in Z}))
    P = pd.DataFrame(rows); C = pd.DataFrame(cons)
    P.to_csv(os.path.join(A.T, "c9a3_pairs.tsv"), sep="\t", index=False); C.to_csv(os.path.join(A.T, "c9a3_consensus.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(P.round(3).to_string(index=False)); print(C.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
