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
    # EVERY OTHER GLOBAL, carried chunk-to-chunk. Previously only psi/pi/alpha_pi were carried, so
    # bb_conc, a, b and the four Beta hyperparameters were re-initialised from their priors on every
    # chunk and given only num_epochs_later steps on ONE batch to move. They never accumulated
    # evidence over the full dataset. On the mouse atlas this left bb_conc bit-identical across arms
    # with the same seed (i.e. still at its initialisation) while psi genuinely differed.
    # CARRY ONLY bb_conc.
    #
    # MEASURED, do not "improve" this without re-measuring. Carrying the per-junction a/b (and their
    # hyperparameters) COMPOUNDS instead of accumulating: each chunk is seeded from the previous
    # chunk's posterior and then re-applies its own likelihood in the same direction. On a 6,000-cell
    # subset of the real atlas, 3 passes drove the psi prior to Beta(204, 6) -- mean 0.97, a+b=210 --
    # which pushed 42% of psi above 0.9, HALVED psi_contrast (0.115 -> 0.059) and collapsed per-cell
    # phi commitment (0.47 -> 0.14). Same pathology as carrying q(psi)'s SCALE, which made the
    # posterior 2.4x too wide rather than tighter.
    #
    # bb_conc is different: a single scalar with no hierarchical structure above it, so it converges
    # rather than compounding -- and it is the one global that is demonstrably STUCK when not carried
    # (bit-identical across arms at the same seed).
    #
    # NB a/b/dir_conc are therefore still LAST-CHUNK estimates. That is a real limitation and it is
    # NOT fixed by carrying them. If a full-data estimate is needed, fit them post hoc with psi frozen.
    CARRY = ("bb_conc",)
    gcarry = {}
    # THE PER-CELL PHI STORE. This is the thing that must not be dropped.
    all_phi = (np.zeros((N, K), dtype=np.float32) if phi_init is None
               else np.asarray(phi_init, dtype=np.float32).copy())
    phi_key = f"phi_init_{K}_waypoints"
    psi_key = f"psi_init_{K}_waypoints"
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

            # CRITICAL, and easy to get wrong: when waypoints_use=True, train() IGNORES the
            # psi_init/phi_init ARGUMENTS and re-reads adata.varm[psi_key] / adata.obsm[phi_key]
            # instead (model.py: "if self.waypoints_use: ... psi_init = varm[psi_key].T"). So passing
            # psi_init=psi_w does NOTHING under waypoint init -- every chunk silently restarts from
            # the ORIGINAL waypoint psi and the chunk-to-chunk warm start is thrown away.
            #
            # run_training.py works around this by writing the CARRIED psi back into the waypoint key
            # (run_training.py:392, `batch_adata.varm[...] = global_psi.T`). Mirror that exactly.
            #
            # Getting this wrong cost a real result: the benchmark reported waypoints+chunked at
            # psi_r 0.724 / ARI 0.494 -- looking like the ATLAS PRODUCTION CONFIG was broken -- when
            # in fact only the benchmark was.
            if psi_w is not None and model_kwargs.get("waypoints_use"):
                ba.varm[psi_key] = np.asarray(psi_w, dtype=np.float32).T      # varm is (J, K)

            model = LeafletFA(
                adata=ba, K=K,
                num_epochs=num_epochs_first if first else num_epochs_later,
                obs_scale=N / len(idx),                       # <- unbiased full-data ELBO
                lr=lr * (lr_pass_decay ** pass_idx),
                pi_init=None if pi_w is None else torch.as_tensor(pi_w, dtype=torch.float32),
                alpha_pi_init=(None if alpha_w is None
                               else torch.as_tensor(alpha_w, dtype=torch.float32)),
                global_init=gcarry or None,
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
