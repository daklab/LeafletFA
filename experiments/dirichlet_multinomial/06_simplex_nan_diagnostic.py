#!/usr/bin/env python
"""Find what diverges when psi_parameterization='simplex' runs at atlas scale.

Run: python 06_simplex_nan_diagnostic.py --cells 4000 --steps 2000 --K 20

Symptom
-------
Two atlas-scale simplex fits died with a non-finite loss at almost the same point -- step 1143
in the first attempt and step 1161 in the second, on different data subsets. The first was
blamed on a global-max softmax (which underflows every other event's denominator to zero); that
bug is fixed and the failure still happens, so the explanation was wrong or incomplete.

What it is not:
  * not the Beta path      -- baseline fits on the same data are healthy at 5+ hours
  * not gradient explosion -- ClippedAdam already runs with clip_norm=10.0
  * not scale-independent  -- simulation runs (K=5, 12,804 junctions) never diverge

So it is specific to the parameterisation AND to scale: K=20 over 79,768 junctions is 1.6M
psi_logit parameters.

What this logs every --every steps
----------------------------------
  loss                to locate the step where it turns
  max|psi_logit|      the softmax is invariant to adding a constant within an event, so the
  within-ATSE spread  logits carry a gauge freedom the likelihood cannot see. If the guide's
                      location drifts along it, magnitudes grow without changing psi at all,
                      until exp() overflows. Leading hypothesis.
  guide scale min/max an AutoNormal scale collapsing to 0 sends log(scale) in the KL to -inf
  psi min, nonfinite  whether psi itself goes bad before the loss does

Reading it
----------
If the within-ATSE SPREAD stays bounded while max|psi_logit| grows, the gauge direction is the
cause, and the fix is to centre logits per event (subtract the segment mean) -- which changes
nothing about psi but removes the unidentified direction. If the SPREAD grows instead, the
likelihood is genuinely driving a junction to 0 or 1 and the fix is a tighter clamp or lower lr.
"""

import argparse
import sys
import time

import h5py
import numpy as np
import scipy.sparse as sp
import torch
import pyro

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.minibatch import LeafletFAMinibatch  # noqa: E402

WD = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
H5 = f"{WD}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad"


def cat(name, which):
    with h5py.File(H5, "r") as f:
        n = f[which][name]
        if isinstance(n, h5py.Group):
            c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
            return c[n["codes"][:]]
        return np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])


def read_rows(layer, rows, n_cols, chunk=2000):
    order = np.argsort(rows)
    srt = np.asarray(rows)[order]
    out = np.zeros((len(rows), n_cols), dtype=np.float32)
    with h5py.File(H5, "r") as f:
        g = f["layers"][layer]
        ip_all = g["indptr"]
        for a in range(0, len(srt), chunk):
            b = min(a + chunk, len(srt))
            blk = srt[a:b]
            lo, hi = int(blk[0]), int(blk[-1]) + 1
            ip = ip_all[lo:hi + 1]
            s, e = int(ip[0]), int(ip[-1])
            m = sp.csr_matrix((g["data"][s:e], g["indices"][s:e], ip - ip[0]),
                              shape=(hi - lo, n_cols))
            out[order[a:b]] = m[blk - lo].toarray()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ev = cat("event_id", "var")
    J = len(ev)
    rng = np.random.default_rng(0)
    with h5py.File(H5, "r") as f:
        N_all = int(f["layers"]["cell_by_junction_matrix"].attrs["shape"][0])
    cells = np.sort(rng.choice(N_all, size=min(args.cells, N_all), replace=False))
    print(f"[1] {len(cells):,} cells x {J:,} junctions, K={args.K}, "
          f"{len(np.unique(ev)):,} ATSEs", flush=True)
    junc = read_rows("cell_by_junction_matrix", cells, J)
    clus = read_rows("cell_by_cluster_matrix", cells, J)
    print(f"[2] loaded ({junc.nbytes/1e9:.1f} GB each)", flush=True)

    pyro.clear_param_store()
    m = LeafletFAMinibatch(junc, clus, K=args.K, device="cpu",
                           psi_parameterization="simplex", event_id=ev)

    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import ClippedAdam
    pyro.set_rng_seed(args.seed)
    # Replicate fit()'s convention exactly: the batch size is stored on the model and used as the
    # plate's default, the guide is built for that batch, and svi.step() takes NO arguments.
    # Passing subsample_size to step() forwards it to the guide, which does not accept it.
    m._B = min(int(args.batch), m.N)
    m.guide = m._make_guide(m._B)
    lrd = args.gamma ** (1.0 / max(args.steps, 1))
    svi = SVI(m.model, m.guide, ClippedAdam({"lr": args.lr, "lrd": lrd, "clip_norm": 10.0}),
              Trace_ELBO())

    st = pyro.get_param_store()
    print(f"\n{'step':>6} {'loss':>14} {'max|logit|':>11} {'spread':>9} "
          f"{'scale_min':>10} {'scale_max':>10} {'psi_min':>9} {'nonfin':>7}", flush=True)
    t0 = time.time()
    diverged = False
    for i in range(args.steps):
        loss = svi.step()
        if i % args.every == 0 or not np.isfinite(loss):
            with torch.no_grad():
                lg = st["AutoNormal.locs.psi_logit"].detach()
                sc = st["AutoNormal.scales.psi_logit"].detach()
                idx = m.atse_idx.expand(lg.shape[0], -1)
                hi = torch.full((lg.shape[0], m.n_atse), -1e30).scatter_reduce(
                    1, idx, lg, reduce="amax", include_self=True)
                lo = torch.full((lg.shape[0], m.n_atse), 1e30).scatter_reduce(
                    1, idx, lg, reduce="amin", include_self=True)
                spread = float((hi - lo).max())
                psi = m.psi_mean()
                nf = int((~torch.isfinite(lg)).sum() + (~torch.isfinite(psi)).sum())
            print(f"{i:>6d} {loss:>14.4e} {float(lg.abs().max()):>11.3f} {spread:>9.3f} "
                  f"{float(sc.min()):>10.4f} {float(sc.max()):>10.4f} "
                  f"{float(psi.min()):>9.2e} {nf:>7d}", flush=True)
        if not np.isfinite(loss):
            print(f"\n>>> NON-FINITE LOSS at step {i} after {time.time()-t0:.0f}s", flush=True)
            diverged = True
            break
    if not diverged:
        print(f"\n>>> completed {args.steps} steps cleanly in {time.time()-t0:.0f}s -- did not "
              f"reproduce at this scale", flush=True)


if __name__ == "__main__":
    main()
