"""Check 1 verification (b) (CPU): on data simulated from the fitted TMS model, the TRUE parameters must
score above the population null (i) fitted on simulated TRAIN cells. If they do not, the scorer or the
null fitting is wrong and no check 1 number is interpreted.

    python c1_verify_b.py --ds TMS

Truth = the seed-0 fit (psi, per-ATSE concentration); 5,000 TRAIN cells (fitted phi) and 2,000 TEST
cells (E-step phi) keep their observed ATSE totals. Deviation from the pre-registration: null (iii)
(K = 1) is not refitted on the simulation; its maximum-likelihood counterpart is null (i) with an ML
concentration, which is what is scored here.
"""
import argparse, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="TMS", choices=("TMS", "Allen"))
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=2000)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    P = vlib.paths(a.ds)
    run = os.path.join(vlib.B, a.ds, "full_K20_s0_f64")
    z = np.load(os.path.join(run, "fit", "fit.npz"), allow_pickle=True)
    psi, conc = z["psi_mean"].astype(np.float64), z["bb_conc_vec"].astype(np.float64)
    _, ev = vlib.read_var(P["train"]); _, inv = vlib.event_index(ev)
    tr_cells = vlib.read_cells(P["train"])
    assert np.array_equal(z["cell_id"].astype(str), tr_cells)
    rng = np.random.default_rng(1)
    rt = np.sort(rng.choice(len(tr_cells), a.n_train, replace=False))
    Yt = vlib.read_counts(P["train"], rt); phit = z["phi"][rt].astype(np.float64)
    te_cells = vlib.read_cells(P["test"])
    f = np.load(os.path.join(run, "factors_real.npz"), allow_pickle=True)
    assert np.array_equal(f["cell_id"].astype(str), te_cells)
    rs = np.sort(rng.choice(len(te_cells), a.n_test, replace=False))
    Ys = vlib.read_counts(P["test"], rs); phis = f["phi"][rs].astype(np.float64)

    St = vlib.simulate(Yt, inv, phit, psi, conc, seed=11)
    Ss = vlib.simulate(Ys, inv, phis, psi, conc, seed=12)
    pop = vlib.usage(St, inv)[0]
    pop = np.where(np.isfinite(pop), pop, 1.0)
    import scipy.sparse as sp
    S = sp.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)))
    pop = pop / np.asarray(pop @ S).ravel()[inv]
    cpop = vlib.fit_conc(St, inv, lambda s: torch.as_tensor(pop)[None, :].expand(s.stop - s.start, -1))
    lt, nt, _ = vlib.score(Ss, inv, vlib.mu_from(phis, psi), conc)
    ln, nn, _ = vlib.score(Ss, inv, vlib.mu_from(np.ones((a.n_test, 1)), pop[None]), cpop)
    assert np.allclose(nt, nn)
    gain = (lt.sum() - ln.sum()) / nt.sum()
    frac_cells = float((lt > ln).mean())
    res = pd.DataFrame([dict(ds=a.ds, n_sim_train=a.n_train, n_sim_test=a.n_test, reads=nt.sum(),
                             truth_per_read=lt.sum() / nt.sum(), null_i_per_read=ln.sum() / nt.sum(),
                             gain_per_read=gain, frac_cells_truth_better=frac_cells,
                             passed=bool(gain > 0))])
    res.to_csv(os.path.join(vlib.V, "tsv", f"c1_verify_b_{a.ds}.tsv"), sep="\t", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
