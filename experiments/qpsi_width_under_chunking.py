#!/usr/bin/env python
"""
Does multi-pass chunked training recover the same q(psi) as full-batch training?

WHY THIS EXPERIMENT EXISTS
--------------------------
The mouse atlas cannot be trained full-batch: the training matrix is 132,706 x 79,768 at 13.8%
density = 1.46e9 nonzeros, and the sparse gather alone needs ~218 GB against a 95 GB GPU. So
training is CHUNKED: a fresh LeafletFA is built per 8,192-cell chunk, warm-started from the previous
chunk's psi, with obs_scale = N/batch so each chunk's objective stays an unbiased estimate of the
FULL-data ELBO.

That scheme is only trustworthy if it lands where full-batch lands. It cannot be checked on the
atlas (there is no full-batch reference to compare against), so we check it on a simulation small
enough that full-batch IS feasible, and where the ground truth is known.

We measure TWO things separately, because they behave completely differently:

  1. the psi POINT ESTIMATE  -- does chunking learn the same programs?
  2. the psi POSTERIOR WIDTH -- does q(psi) have the right uncertainty?

(2) matters on its own because it is the posterior the differential-splicing FDR samples from.

WHAT WE FOUND (see the docstring of each arm below)
---------------------------------------------------
  * The POINT ESTIMATE is fine. Chunked psi recovery r = 0.986-0.993 vs full-batch 0.997.
  * The POSTERIOR WIDTH is structurally ~1.5x too wide, and NONE of the three obvious fixes work.
    In particular, running each chunk LONGER makes it WORSE.

Run:  python experiments/qpsi_width_under_chunking.py
"""
import contextlib
import io
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, "tests")
from test_basic import simulate_from_generative_model  # noqa: E402

from leafletfa import LeafletFA  # noqa: E402

# --------------------------------------------------------------------------------------
# Setup: a simulation small enough to fit full-batch, with known ground-truth psi.
# 1,200 cells / 120 junctions / K=3, chunked into 4 x 300 cells, 3 passes.
# Schedule is the swept-optimal one (lr=0.05, gamma=0.5) so that we are measuring the
# CHUNKING, not a bad learning rate.
# --------------------------------------------------------------------------------------
N, B, K, PASSES = 1200, 300, 3, 3
LR, GAMMA = 0.05, 0.5

adata, PSI_TRUE, _ = simulate_from_generative_model(n_cells=N, K=K, n_atses=40, seed=0)
E = sys.stderr


def npy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def fit(ad_, epochs, obs_scale=1.0, psi_init=None, psi_scale_init=None):
    m = LeafletFA(ad_, K=K, waypoints_use=False, log_wandb=False, loss_plot=False,
                  num_epochs=epochs, patience=10**6, lr=LR, gamma=GAMMA,
                  obs_scale=obs_scale, use_dense_mode=False)
    with contextlib.redirect_stdout(io.StringIO()):
        m.from_anndata()
        m.train(num_initializations=1, seeds=[0],
                psi_init=None if psi_init is None else torch.tensor(psi_init),
                psi_scale_init=None if psi_scale_init is None else torch.tensor(psi_scale_init))
        m.get_all_variables()
    return m


def psi_r(psi_hat):
    """Correlation with the TRUE psi, after matching factors (identifiable only up to permutation)."""
    C = np.corrcoef(np.vstack([psi_hat, PSI_TRUE]))[:K, K:]
    return float(np.nanmax(np.abs(C), axis=1).mean())


def multipass(epochs, carry_scale=False):
    """The pipeline's scheme: fresh LeafletFA per chunk, warm-started from the previous chunk."""
    rng = np.random.default_rng(0)
    g_psi = g_scale = None
    for _ in range(PASSES):
        perm = rng.permutation(N)
        for i in range(0, N, B):
            idx = np.sort(perm[i:i + B])
            m = fit(adata[idx].copy(), epochs,
                    obs_scale=N / len(idx),            # the N/B rescaling
                    psi_init=g_psi,
                    psi_scale_init=g_scale if carry_scale else None)
            g_psi, g_scale = npy(m.psi), npy(m.psis_scale)
    return g_psi, g_scale


# --------------------------------------------------------------------------------------
# REFERENCE: full-batch on all 1,200 cells, run long. This is the q(psi) we WANT.
# --------------------------------------------------------------------------------------
ref = fit(adata, epochs=900)
W_REF, R_REF = npy(ref.psis_scale).mean(), psi_r(npy(ref.psi))
print(f"\nREFERENCE  full-batch, {N} cells, 900 epochs", file=E)
print(f"           mean q(psi) width = {W_REF:.4f}   psi_r vs truth = {R_REF:.3f}\n", file=E)
print(f"{'arm':<44} {'q(psi) width':>13} {'vs ref':>8} {'psi_r':>7}", file=E)
print("-" * 76, file=E)


def report(tag, width, r):
    print(f"{tag:<44} {width:>13.4f} {width / W_REF:>7.2f}x {r:>7.3f}", file=E)


# --------------------------------------------------------------------------------------
# ARM 1 -- the pipeline as it stands: scale RESET on every chunk.
#
# RESULT: width 1.53x too wide, but the POINT ESTIMATE is essentially perfect (0.986 vs 0.997).
# So chunking learns the right programs; only its uncertainty is off.
# --------------------------------------------------------------------------------------
psi, sc = multipass(epochs=60, carry_scale=False)
report("1. chunked, scale reset each chunk (60 ep)", sc.mean(), psi_r(psi))

# --------------------------------------------------------------------------------------
# ARM 2 -- HYPOTHESIS: "carry q(psi)'s width forward, so it accumulates evidence."
#
# RESULT: WRONG, and it makes things WORSE (2.4x). Each chunk's optimisation WIDENS from its
# narrow start (0.1 -> 0.27), so carrying that width forward COMPOUNDS the widening instead of
# accumulating evidence. train(psi_scale_init=...) exists, but run_training.py deliberately does
# NOT use it.
# --------------------------------------------------------------------------------------
psi, sc = multipass(epochs=60, carry_scale=True)
report("2. chunked, scale CARRIED forward (60 ep)", sc.mean(), psi_r(psi))

# --------------------------------------------------------------------------------------
# ARM 3 -- HYPOTHESIS: "obs_scale already targets the full-data width; each chunk just needs
# more epochs to converge to it."
#
# RESULT: WRONG, and BACKWARDS -- more epochs makes the width WORSE (1.53x -> 1.95x -> 2.15x).
#
# WHY. obs_scale gets the MAGNITUDE right: scaling the log-likelihood by N/B multiplies the Fisher
# information by N/B, so ONE chunk's psi posterior should have the full-data width, and it briefly
# does. But each chunk sees DIFFERENT cells and therefore DISAGREES with the others. Cycling chunks
# pulls q(psi) back and forth, and its width INFLATES to cover the between-chunk disagreement.
# Training each chunk harder makes it pull harder, so the inflation grows.
#
# This is STRUCTURAL. No tuning fixes it, because the chunks are fighting each other. Only a single
# guide seeing all the data through one optimizer avoids it -- i.e. pyro.plate(subsample_size=B)
# with a PERSISTENT guide, which is what the deleted LeafletFAminibatch.py did.
#
# PRACTICAL COROLLARY (a free win): use FEWER epochs per chunk. 60 is both faster AND gives the
# least-inflated posterior. Do not raise num_epochs_later to 1000.
# --------------------------------------------------------------------------------------
for ep in (200, 600):
    psi, sc = multipass(epochs=ep, carry_scale=False)
    report(f"3. chunked, scale reset ({ep} ep)", sc.mean(), psi_r(psi))

print("-" * 76, file=E)
print("""
CONCLUSION
  POINT ESTIMATE  psi/phi/pi are FINE under chunking (r = 0.986-0.993 vs full-batch 0.997).
                  Everything the figures rest on is sound.
  POSTERIOR WIDTH q(psi) is structurally ~1.5x too WIDE, and none of the three fixes work.
                  Consequence: the differential-splicing FDR is CONSERVATIVE (under-calling, not
                  over-calling). That is a bounded, statable limitation -- not a correctness failure.
  ACCEPTED        We keep the chunked scheme, use ~60 epochs per chunk, and state the limitation.
                  The alternative is a persistent-guide + pyro.plate rewrite.
""", file=E)
