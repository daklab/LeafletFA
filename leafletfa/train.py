"""Train LeafletFA with the shipped recipe on a model-ready h5ad.

    python -m leafletfa.train --h5 <model_ready.h5ad> --K 20 --seed 0 --out <dir>

The recipe (settled on the mouse atlas, 2026-08): Dirichlet-multinomial likelihood over whole
ATSEs, within-ATSE simplex psi, per-ATSE concentration floored at 2, Dirichlet concentration on
phi pinned at K, 16,200 steps x 8,192-cell minibatches, lr 0.025 with exponential decay to 0.1x,
no early stopping (the ELBO is noise-dominated at this scale; convergence is judged on the
psi-contrast trace). For datasets under ~60k cells at K = 30 use --lr 0.015: the larger rank at
the higher rate diverges repeatedly there (measured 2026-08-18).

This is the packaged, self-contained driver. The production analysis driver with resumable
checkpoint files, wandb, holdout splits and per-run reports is A44_train_atlas_dm.py in the
analysis repository; both call the same LeafletFAMinibatch.fit(), which carries the NaN
rollback guard.
"""
from __future__ import annotations
import argparse, json, os, time

import numpy as np


def _cat(g):
    import h5py
    if isinstance(g, h5py.Group):
        c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]])
        return c[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def _layer(f, name):
    import scipy.sparse as sp
    g = f["layers"][name]
    shape = tuple(g.attrs["shape"]) if "shape" in g.attrs else None
    return sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="model-ready h5ad with cell_by_junction_matrix, "
                                                "cell_by_cluster_matrix layers and var.event_id")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=16200)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--conc-min", type=float, default=2.0)
    ap.add_argument("--max-nan-restarts", type=int, default=40)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    import h5py
    from leafletfa.minibatch import LeafletFAMinibatch

    t0 = time.time()
    with h5py.File(args.h5, "r") as f:
        junc = _layer(f, "cell_by_junction_matrix").toarray().astype("float32")
        clus = _layer(f, "cell_by_cluster_matrix").toarray().astype("float32")
        ev = _cat(f["var"]["event_id"])
    print(f"[load] {junc.shape[0]:,} cells x {junc.shape[1]:,} junctions, "
          f"{len(np.unique(ev)):,} ATSEs ({time.time()-t0:.0f}s)", flush=True)

    m = LeafletFAMinibatch.dm(junc, clus, K=args.K, event_id=ev,
                              conc_min=args.conc_min, device=args.device)
    try:
        import torch
        m.fixed_dir_conc = torch.tensor(float(args.K), device=m.device)  # delta = K, the shipped pin
    except Exception:
        pass
    m.fit(num_steps=args.steps, subsample_size=args.batch, lr=args.lr, gamma=args.gamma,
          seed=args.seed, early_stop=False, max_nan_restarts=args.max_nan_restarts,
          checkpoint_every=200, verbose=True)

    os.makedirs(args.out, exist_ok=True)
    run = f"dm_simplex_K{args.K}_seed{args.seed}_b{args.batch}"
    d = os.path.join(args.out, run); os.makedirs(d, exist_ok=True)
    psi = m.psi_mean().detach().cpu().numpy()
    phi = m.phi_mean().detach().cpu().numpy()
    loc, scale = m.psi_loc_scale()
    np.savez_compressed(os.path.join(d, "fit.npz"),
                        psi_mean=psi.astype("float32"), phi=phi.astype("float32"),
                        psi_loc=(loc.detach().cpu().numpy().astype("float32") if loc is not None else np.zeros(0, "float32")),
                        psi_scale=(scale.detach().cpu().numpy().astype("float32") if scale is not None else np.zeros(0, "float32")),
                        event_id=np.unique(ev), inv_ev=np.unique(ev, return_inverse=True)[1],
                        K=args.K, seed=args.seed, steps_run=m.stopped_step,
                        losses=np.asarray(m.losses, "float32"))
    with open(os.path.join(d, "result.json"), "w") as fh:
        json.dump(dict(run=run, K=args.K, seed=args.seed, steps_run=int(m.stopped_step),
                       n_nan_restarts=int(m.n_nan_restarts), lr=args.lr,
                       train_minutes=round((time.time() - t0) / 60, 1)), fh, indent=2)
    print(f"[done] {d} ({(time.time()-t0)/60:.0f} min, {m.n_nan_restarts} NaN restarts)", flush=True)


if __name__ == "__main__":
    main()
