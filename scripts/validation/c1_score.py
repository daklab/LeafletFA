"""Check 1 (and 2), scoring step (CPU): held-out log-likelihood per observed read on TEST.

    python c1_score.py --ds TMS --run <dir with fit/fit.npz, factors_A.npz, factors_B.npz, factors_real.npz>
                       --k1 <dir with fit/fit.npz of the K = 1 model> --tag K20_s0

Model, event split: score(B | phi_A) + score(A | phi_B); naive: score(all | phi_real) (optimistic).
Nulls fitted on TRAIN (c1_nulls.py): (i) population psi, (ii) per-label psi, (iii) K = 1 model;
(ii-shuffled) for verification (c). Gains per read with 95 % CIs from 1,000 bootstraps over mice.
Writes tsv/c1_<ds>_<tag>_scores.tsv (one row per scorer) and tsv/c1_<ds>_<tag>_gains.tsv.
"""
import argparse, os, sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402


def phi_of(run, name, cells):
    f = np.load(os.path.join(run, f"factors_{name}.npz"), allow_pickle=True)
    pos = {c: i for i, c in enumerate(f["cell_id"].astype(str))}
    return f["phi"][[pos[c] for c in cells]].astype(np.float64), int(f["n_nan_restarts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=("TMS", "Allen"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--k1", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    P = vlib.paths(a.ds)
    C1 = os.path.join(vlib.V, "check1", a.ds)
    _, ev = vlib.read_var(P["test"]); _, inv = vlib.event_index(ev)
    cells, lab, mouse = vlib.obs_labels(P["test"])
    Y = vlib.read_counts(P["test"])
    A = sp.load_npz(os.path.join(C1, "test_A.npz")).tocsr(); Bm = sp.load_npz(os.path.join(C1, "test_B.npz")).tocsr()

    z = np.load(os.path.join(a.run, "fit", "fit.npz"), allow_pickle=True)
    psi, conc = z["psi_mean"].astype(np.float64), z["bb_conc_vec"].astype(np.float64)
    phiA, rA = phi_of(a.run, "A", cells); phiB, rB = phi_of(a.run, "B", cells)
    per = {}
    lB, nB, _ = vlib.score(Bm, inv, vlib.mu_from(phiA, psi), conc)
    lA, nA, _ = vlib.score(A, inv, vlib.mu_from(phiB, psi), conc)
    per["model_split"] = (lA + lB, nA + nB)
    if os.path.exists(os.path.join(a.run, "factors_real.npz")):
        phiR, _ = phi_of(a.run, "real", cells)
        per["model_naive"] = vlib.score(Y, inv, vlib.mu_from(phiR, psi), conc)[:2]

    nl = np.load(os.path.join(C1, "nulls.npz"), allow_pickle=True)
    pop = nl["pop"].astype(np.float64)
    per["null_i_population"] = vlib.score(Y, inv, vlib.mu_from(np.ones((len(cells), 1)), pop[None]),
                                          nl["conc_pop"])[:2]
    labs = list(nl["labels"].astype(str)); kept = nl["label_kept"]
    li = np.array([labs.index(l) if (l in labs and kept[labs.index(l)]) else -1 for l in lab])
    U = np.vstack([nl["psi_lab"].astype(np.float64), pop[None]])            # row -1 -> population
    Ush = np.vstack([nl["psi_lab_shuffled"].astype(np.float64), pop[None]])
    li_u = np.where(li >= 0, li, len(labs))
    per["null_ii_celltype"] = vlib.score(Y, inv, vlib.mu_label(li_u, U), nl["conc_lab"])[:2]
    per["null_ii_shuffled_labels"] = vlib.score(Y, inv, vlib.mu_label(li_u, Ush), nl["conc_lab"])[:2]
    if a.k1:
        z1 = np.load(os.path.join(a.k1, "fit", "fit.npz"), allow_pickle=True)
        per["null_iii_K1"] = vlib.score(Y, inv, vlib.mu_from(np.ones((len(cells), 1)), z1["psi_mean"].astype(np.float64)),
                                        z1["bb_conc_vec"])[:2]

    reads = per["null_i_population"][1]
    for k, (l, n) in per.items():                       # every scorer covers the same reads
        assert np.allclose(n, reads), k
    rows = [dict(ds=a.ds, tag=a.tag, scorer=k, total_loglik=l.sum(), reads=n.sum(),
                 loglik_per_read=l.sum() / n.sum(), n_cells=len(cells),
                 test_labels_not_in_train=int((li < 0).sum()) if "null_ii" in k else None)
            for k, (l, n) in per.items()]
    sc = pd.DataFrame(rows)
    sc["estep_restarts_A_B"] = f"{rA},{rB}"

    um, mi = np.unique(mouse, return_inverse=True)
    M = sp.csr_matrix((np.ones(len(mi)), (mi, np.arange(len(mi)))), shape=(len(um), len(mi)))
    rng = np.random.default_rng(20261025)
    boots = rng.integers(0, len(um), (a.n_boot, len(um)))
    gains = []
    lm = per["model_split"][0]
    for k in [x for x in per if x.startswith("null")] + (["model_naive"] if "model_naive" in per else []):
        d = M @ (lm - per[k][0]); r = M @ reads
        est = d.sum() / r.sum()
        bs = np.array([d[b].sum() / r[b].sum() for b in boots])
        gains.append(dict(ds=a.ds, tag=a.tag, comparison=f"model_split - {k}", gain_per_read=est,
                          ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), n_mice=len(um)))
    gn = pd.DataFrame(gains)
    os.makedirs(os.path.join(vlib.V, "tsv"), exist_ok=True)
    sc.to_csv(os.path.join(vlib.V, "tsv", f"c1_{a.ds}_{a.tag}_scores.tsv"), sep="\t", index=False)
    gn.to_csv(os.path.join(vlib.V, "tsv", f"c1_{a.ds}_{a.tag}_gains.tsv"), sep="\t", index=False)
    print(sc[["scorer", "loglik_per_read", "reads"]].to_string(index=False))
    print(gn.to_string(index=False))


if __name__ == "__main__":
    main()
