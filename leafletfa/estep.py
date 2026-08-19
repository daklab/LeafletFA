"""Frozen-global E-step through the current (plate minibatch) implementation.

`04e` and `07c` infer phi for held-out cells against a trained dictionary. Both did this through the
legacy `LeafletFA` class, i.e. a different implementation from the one that trained the models. On
the simulation, the legacy trainer converged to the right answer on only ~1/3 of runs (Psi recovery
ranged 0.40-0.99, sd 0.18) whereas the plate implementation was consistent (sd 0.001). The E-step is
an easier problem than training — with psi frozen the per-cell problems are independent and
K-dimensional — but there is no reason to keep two implementations in the pipeline, so this routes
the E-step through the same code that produced the models.

Everything shared across cells is frozen: psi, pi, alpha_pi, dir_conc and the Beta-Binomial
concentration. Freezing psi alone would not be enough — pi and dir_conc together are the Dirichlet
prior on phi, so re-learning them on the held-out cells would infer phi under a different prior than
the training cells saw, confounding exactly the train-vs-heldout comparison these scripts measure.

Usage (drop-in for the existing `infer_phi_chunk` body):

    from leafletfa.estep import infer_phi_minibatch
    phi = infer_phi_minibatch(chunk_adata, psi, K, dir_conc=..., pi=..., alpha_pi=...,
                              bb_conc=..., junc_prior=..., num_epochs=..., device=...)
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

_JUNC_LAYER = "cell_by_junction_matrix"
_CLUS_LAYER = "cell_by_cluster_matrix"


def _dense(x):
    return np.asarray(x.todense() if sp.issparse(x) else x)


def infer_phi_minibatch(chunk_adata, psi, K, dir_conc=None, pi=None, alpha_pi=None,
                        bb_conc=None, junc_prior=True, num_epochs=3000, lr=0.1, gamma=0.1,
                        seed=0, device=None, subsample_size=None, verbose=False,
                        likelihood=None, event_id=None, conc_min=1e-3, dm_alpha_floor=1e-2):
    """Infer phi for `chunk_adata` against a frozen dictionary. Returns (n_cells, K) float32.

    `psi` must be (K, n_vars) and aligned to `chunk_adata.var_names` — the caller is responsible for
    that alignment, exactly as with the previous implementation.

    likelihood
        None keeps the legacy behaviour: binomial if bb_conc is inf, else betabinomial. Pass
        "dirichletmultinomial" for the shipped atlas model. That model needs `event_id` (so the
        ATSE index can be built) and takes `bb_conc` as a PER-ATSE vector -- the concentration each
        event was trained with, not a scalar. Under the DM, psi is a within-ATSE composition and
        is frozen as given; the model does not sigmoid or renormalise it.
    """
    import torch
    import pyro
    import sys
    sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
    from leafletfa.minibatch import LeafletFAMinibatch

    psi = np.asarray(psi, dtype=np.float32)
    if psi.shape[1] != chunk_adata.n_vars:
        raise ValueError(f"psi has {psi.shape[1]} junctions but the chunk has "
                         f"{chunk_adata.n_vars}; alignment is the caller's responsibility")

    junc = _dense(chunk_adata.layers[_JUNC_LAYER]).astype(np.float32)
    clus = _dense(chunk_adata.layers[_CLUS_LAYER]).astype(np.float32)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pyro.clear_param_store()
    if likelihood == "dirichletmultinomial":
        if event_id is None:
            raise ValueError("the Dirichlet-multinomial E-step needs event_id to index ATSEs")
        conc = np.asarray(bb_conc, dtype=np.float32) if bb_conc is not None else None
        m = LeafletFAMinibatch(junc, clus, K=int(K), device=device,
                               likelihood="dirichletmultinomial",
                               psi_parameterization="simplex", event_id=np.asarray(event_id),
                               conc_per_atse=(conc is not None and conc.ndim == 1),
                               conc_min=conc_min, dm_alpha_floor=dm_alpha_floor)
        m.freeze_globals(psi=psi, pi=pi, alpha_pi=alpha_pi, dir_conc=dir_conc, bb_conc=conc)
    else:
        m = LeafletFAMinibatch(junc, clus, K=int(K), junc_specific_prior=bool(junc_prior),
                               device=device,
                               likelihood="binomial" if _is_inf(bb_conc) else "betabinomial")
        m.freeze_globals(psi=psi, pi=pi, alpha_pi=alpha_pi, dir_conc=dir_conc,
                         bb_conc=None if _is_inf(bb_conc) else bb_conc)
    # checkpoint_every must divide the budget finely enough that early stopping can actually fire:
    # fit() stops after `patience` (8) consecutive non-improving checkpoints, so with the default
    # interval of 100 the soonest possible stop is 800 steps -- unreachable within the 500-step
    # E-step budget, meaning every chunk ran to the cap regardless of convergence. At 25 there are
    # 20 checks in 500 steps, so a converged chunk stops at ~200.
    ckpt = max(5, int(num_epochs) // 20)
    phi = m.infer_phi(num_steps=int(num_epochs), subsample_size=subsample_size,
                      lr=lr, gamma=gamma, seed=seed, verbose=verbose, checkpoint_every=ckpt)
    info = dict(steps_used=int(getattr(m, "stopped_step", num_epochs)),
                stopped_early=bool(getattr(m, "stopped_early", False)))
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(phi, dtype=np.float32), info


def _is_inf(x):
    if x is None:
        return False
    try:
        return not np.isfinite(float(x))
    except (TypeError, ValueError):
        return str(x).lower() in ("inf", "infinity")
