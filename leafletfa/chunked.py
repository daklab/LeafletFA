"""Multi-pass minibatch ("chunked") training — the scheme used to fit the mouse atlas.

WHY THIS MODULE EXISTS
----------------------
This scheme used to live ONLY inside the analysis repo's `run_training.py`. Anything else that needed
it -- the Figure-2 benchmark, tests, a user with a large dataset -- had to hand-copy a 200-line loop.
A hand-copy of a training loop drifts, and when the BENCHMARK's copy drifts, the benchmark silently
stops testing what actually ships.

That is not hypothetical. The Figure-2 notebook's copy dropped the phi carry-over entirely and invented
a final frozen-psi E-step that production never runs, so the full-batch-vs-chunked comparison -- the one
result that justifies chunking for the whole atlas -- was measuring a scheme nobody uses.

So: ONE implementation, called by everything. If you are tempted to re-implement this loop, don't.

WHAT THE SCHEME IS (and the three things people get wrong)
---------------------------------------------------------
Cells are shuffled and split into chunks of `batch_size`; the whole dataset is swept `num_passes` times.

1. GLOBALS (psi, pi, alpha_pi) are warm-started chunk -> chunk. Obvious, and everyone gets it right.

2. PHI IS CARRIED PER CELL.  <-- the one that gets dropped
   After every chunk, each cell's phi is written back into a persistent (n_cells, K) store, and the
   next time a chunk contains that cell, its phi is warm-started from that store. phi is therefore
   CONTINUOUSLY REFINED every time a cell is encountered, across all passes.

3. THE FINAL PHI **IS** THAT ACCUMULATED STORE.  <-- the one that gets invented away
   There is NO final frozen-psi E-step over all cells. `assign_post` is the store itself.

4. `obs_scale = N / B` rescales the LOCAL terms (likelihood + the `assign` prior/entropy) so that each
   chunk's objective is an unbiased estimate of the FULL-data ELBO. Without it, q(psi) has the width it
   would have if the dataset were `batch_size` cells, not `n_cells` -- and that is the posterior the
   differential-splicing FDR then samples from.
"""
import numpy as np
import torch

from .model import LeafletFA


def fit_chunked(adata, K, batch_size=8192, num_passes=20, seed=0,
                num_epochs_first=150, num_epochs_later=60,
                lr=0.05, lr_pass_decay=0.95, phi_init=None, psi_init=None,
                progress=None, **model_kwargs):
    """Multi-pass minibatch training. Returns the final LeafletFA with atlas-wide psi and phi.

    Parameters
    ----------
    adata : AnnData with the two LeafletFA layers.
    batch_size : cells per chunk.
    num_passes : sweeps over the full dataset.
    phi_init, psi_init : optional warm starts (e.g. from waypoints), shapes (n_cells, K) / (K, J).
    lr_pass_decay : lr is multiplied by this ** pass_idx, as run_training.py does.
    progress : optional callable(pass_idx, chunk_idx, n_chunks, model) for logging.
    **model_kwargs : forwarded to LeafletFA (junc_specific_prior, input_conc_prior, gamma, patience...).

    Returns
    -------
    LeafletFA whose `.psi` is the final global psi and whose `.assign_post` is the ACCUMULATED
    per-cell phi (n_cells, K) -- not a re-inferred one.
    """
    N = adata.n_obs
    rng = np.random.default_rng(seed)

    psi_w = None if psi_init is None else np.asarray(psi_init, dtype=np.float32)
    pi_w = alpha_w = None
    # THE PER-CELL PHI STORE. This is the thing that must not be dropped.
    all_phi = (np.zeros((N, K), dtype=np.float32) if phi_init is None
               else np.asarray(phi_init, dtype=np.float32).copy())
    phi_key = f"phi_init_{K}_waypoints"
    model = None

    for pass_idx in range(num_passes):
        perm = rng.permutation(N)
        chunks = [np.sort(perm[i:i + batch_size]) for i in range(0, N, batch_size)]
        for c_idx, idx in enumerate(chunks):
            first = (pass_idx == 0 and c_idx == 0)
            ba = adata[idx].copy()
            seen = all_phi[idx].sum() > 0
            if seen:
                ba.obsm[phi_key] = all_phi[idx]

            model = LeafletFA(
                adata=ba, K=K,
                num_epochs=num_epochs_first if first else num_epochs_later,
                obs_scale=N / len(idx),                       # <- unbiased full-data ELBO
                lr=lr * (lr_pass_decay ** pass_idx),
                pi_init=None if pi_w is None else torch.as_tensor(pi_w, dtype=torch.float32),
                alpha_pi_init=(None if alpha_w is None
                               else torch.as_tensor(alpha_w, dtype=torch.float32)),
                **model_kwargs)
            model.from_anndata()
            model.train(num_initializations=1, seeds=[seed],
                        psi_init=None if psi_w is None else torch.as_tensor(psi_w),
                        phi_init=(torch.as_tensor(all_phi[idx]) if seen else None))
            model.get_all_variables()

            psi_w = _np(model.psi)
            pi_w = _np(model.pi)
            alpha_w = np.asarray(_np(model.alpha_pi), dtype=np.float32).reshape(())
            all_phi[idx] = _np(model.assign_post)             # <- write phi BACK, per cell

            if progress is not None:
                progress(pass_idx, c_idx, len(chunks), model)

    # The final model IS the last chunk's globals + the ACCUMULATED per-cell phi.
    # NOT a fresh frozen-psi E-step. See the module docstring.
    model.psi = psi_w
    model.psi_learned = psi_w
    model.assign_post = all_phi
    model.pi = pi_w
    return model


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
