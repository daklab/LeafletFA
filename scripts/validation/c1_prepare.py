"""Check 1, step 1 (CPU): event-split TEST inputs and scorer verification (a).

    python c1_prepare.py --ds TMS

Writes <V>/check1/<ds>/{test_A.npz, test_B.npz, test_cells.tsv, var.tsv} (A = events kept for the
E-step of half 1, B the complement) and tsv/c1_verify_a_<ds>.tsv: the validation scorer against the
package's dense dm_log_prob plus the multinomial coefficient, on 500 TEST cells with the seed-0 phi.
"""
import argparse, os, sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=("TMS", "Allen"))
    a = ap.parse_args()
    P = vlib.paths(a.ds)
    out = os.path.join(vlib.V, "check1", a.ds); os.makedirs(out, exist_ok=True)
    jid, ev = vlib.read_var(P["test"])
    _, inv = vlib.event_index(ev)
    cells = vlib.read_cells(P["test"])
    Y = vlib.read_counts(P["test"])
    A, Bm = vlib.event_split(Y, inv, vlib.cell_keys(cells))
    assert abs((A + Bm - Y).sum()) < 1e-6 and A.multiply(Bm).nnz == 0
    for name, M in (("A", A), ("B", Bm)):
        p = os.path.join(out, f"test_{name}.npz")
        if not os.path.exists(p):
            sp.save_npz(p, M.tocsr().astype(np.float32))
    pd.DataFrame({"cell_id": cells}).to_csv(os.path.join(out, "test_cells.tsv"), sep="\t", index=False)
    pd.DataFrame({"junction_id": jid, "event_id": ev}).to_csv(os.path.join(out, "var.tsv"), sep="\t", index=False)
    rA, rB = np.asarray(A.sum(1)).ravel(), np.asarray(Bm.sum(1)).ravel()
    print(f"{a.ds}: {Y.shape[0]:,} TEST cells; reads A {rA.sum():,.0f} / B {rB.sum():,.0f}; "
          f"cells with an empty half: A {(rA == 0).sum()}, B {(rB == 0).sum()}", flush=True)

    # verification (a): sparse full-pmf scorer == dense dm_log_prob + multinomial coefficient
    from leafletfa.dirichlet_multinomial import dm_log_prob
    run = os.path.join(vlib.B, a.ds, "full_K20_s0_f64")
    z = np.load(os.path.join(run, "fit", "fit.npz"), allow_pickle=True)
    f = np.load(os.path.join(run, "factors_real.npz"), allow_pickle=True)
    assert np.array_equal(f["cell_id"].astype(str), cells)
    idx = np.sort(np.random.default_rng(0).choice(len(cells), 500, replace=False))
    Ys, phi = Y[idx], f["phi"][idx].astype(np.float64)
    psi, conc = z["psi_mean"].astype(np.float64), z["bb_conc_vec"].astype(np.float64)
    l_sp, r_sp, _ = vlib.score(Ys, inv, vlib.mu_from(phi, psi), conc)
    y = torch.as_tensor(Ys.toarray(), dtype=torch.float64)
    mu = torch.as_tensor(phi @ psi).clamp(vlib.CLIP, 1 - vlib.CLIP)
    it = torch.as_tensor(inv)
    lp = dm_log_prob(y, mu, torch.as_tensor(conc)[it], it, len(np.unique(inv)), eps=vlib.EPS)
    Ne = torch.zeros(y.shape[0], int(inv.max()) + 1, dtype=torch.float64).index_add(1, it, y)
    coef = torch.lgamma(Ne + 1).sum(1) - torch.lgamma(y + 1).sum(1)
    l_dense = (lp + coef).numpy()
    d = np.abs(l_sp - l_dense)
    res = pd.DataFrame([dict(ds=a.ds, n_cells=500, max_abs_diff=d.max(), median_abs_diff=np.median(d),
                             max_rel_diff=(d / np.abs(l_dense)).max(), passed=bool(d.max() < 1e-3))])
    os.makedirs(os.path.join(vlib.V, "tsv"), exist_ok=True)
    res.to_csv(os.path.join(vlib.V, "tsv", f"c1_verify_a_{a.ds}.tsv"), sep="\t", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
