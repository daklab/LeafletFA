"""Check 1, step 2 (CPU): fit nulls (i) population psi and (ii) per-label psi on TRAIN only, each with
a per-ATSE maximum-likelihood concentration (floored at 2), and the label-shuffled (ii) used by
verification (c).

    python c1_nulls.py --ds TMS     -> <V>/check1/<ds>/nulls.npz
"""
import argparse, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402

MIN_CELLS = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=("TMS", "Allen"))
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    t0 = time.time()
    P = vlib.paths(a.ds)
    out = os.path.join(vlib.V, "check1", a.ds, "nulls.npz")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    _, ev = vlib.read_var(P["train"]); _, inv = vlib.event_index(ev)
    cells, lab, mouse = vlib.obs_labels(P["train"])
    assert np.array_equal(cells, vlib.read_cells(P["train"]))
    Y = vlib.read_counts(P["train"])
    print(f"{a.ds} TRAIN {Y.shape[0]:,} cells, nnz {Y.nnz:,}", flush=True)

    pop = vlib.usage(Y, inv)[0]
    pop = np.where(np.isfinite(pop), pop, 1.0)                              # no TRAIN reads: flat
    import scipy.sparse as sp
    S = sp.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)))
    pop = pop / np.asarray(pop @ S).ravel()[inv]

    labs, li = np.unique(lab, return_inverse=True)
    n_per = np.bincount(li)
    keep = n_per >= MIN_CELLS
    U = vlib.usage(Y, inv, li, len(labs))
    U = np.where(np.isfinite(U) & keep[:, None], U, pop[None, :])          # fallbacks to (i)
    print(f"labels {len(labs)} ({keep.sum()} with >= {MIN_CELLS} TRAIN cells); "
          f"label-event entries falling back to (i): {np.mean(~np.isfinite(vlib.usage(Y, inv, li, len(labs)))):.3f}",
          flush=True)

    conc_pop = vlib.fit_conc(Y, inv, lambda s: torch.as_tensor(pop)[None, :].expand(s.stop - s.start, -1))
    print(f"conc (i) median {np.median(conc_pop):.2f}, at floor {(conc_pop <= 2.0001).mean():.2f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    conc_lab = vlib.fit_conc(Y, inv, vlib.mu_label(li, U))
    print(f"conc (ii) median {np.median(conc_lab):.2f}, at floor {(conc_lab <= 2.0001).mean():.2f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # verification (c): labels permuted across TRAIN cells; scored with (ii)'s concentration
    li_sh = np.random.default_rng(20261025).permutation(li)
    Ush = vlib.usage(Y, inv, li_sh, len(labs))
    Ush = np.where(np.isfinite(Ush) & keep[:, None], Ush, pop[None, :])
    np.savez_compressed(out, pop=pop.astype(np.float32), conc_pop=conc_pop.astype(np.float32),
                        labels=labs, n_train_per_label=n_per, label_kept=keep,
                        psi_lab=U.astype(np.float32), conc_lab=conc_lab.astype(np.float32),
                        psi_lab_shuffled=Ush.astype(np.float32))
    print(f"wrote {out} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
