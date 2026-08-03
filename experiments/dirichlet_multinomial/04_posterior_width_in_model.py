#!/usr/bin/env python
"""End-to-end check: does the DM actually widen the psi posterior inside the fitted model?

Run: python 04_posterior_width_in_model.py

Scripts 01 and 02 show the effect analytically, on a single ATSE with psi held fixed. This one
fits the real model three ways on the same simulated data and measures the width of q(psi) that
comes out, which is what the differential-splicing calls are actually drawn from.

Expectation, stated first: the widening should come from the LIKELIHOOD, not the constraint.
The simplex removes an unidentified direction; it does not change how much evidence each event
carries. So betabinomial+simplex should sit near betabinomial+beta, and only the DM should move
the width.

Measured (1200 cells, 40 ATSEs x 3 junctions, K=3, 500 steps):

    betabinomial         + beta        median psi sd  0.00743   1.00x
    betabinomial         + simplex     median psi sd  0.00759   1.02x
    dirichletmultinomial + simplex     median psi sd  0.00957   1.29x

1.29x at m=3 is in line with the analytic DM/BetaBinomial curvature ratio for 3-junction events
in script 02, so the implementation is behaving the way the derivation says it should.
"""

import sys

import numpy as np
import torch
import pyro

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.minibatch import LeafletFAMinibatch  # noqa: E402


def simulate(N=1200, A=40, m=3, K=3, seed=0, mean_cov=15):
    """Counts drawn as one multinomial per (cell, ATSE) from a true composition."""
    rng = np.random.default_rng(seed)
    J = A * m
    event_id = np.repeat(np.arange(A), m)
    psi = rng.dirichlet(np.ones(m), size=(K, A)).reshape(K, J)
    phi = rng.dirichlet(np.ones(K) * 0.5, size=N)
    mu = np.einsum("ck,kj->cj", phi, psi)
    N_e = rng.poisson(mean_cov, size=(N, A)) + 1
    junc = np.zeros((N, J), dtype=np.float32)
    clus = np.zeros((N, J), dtype=np.float32)
    for a in range(A):
        cols = np.arange(a * m, (a + 1) * m)
        p = mu[:, cols] / mu[:, cols].sum(1, keepdims=True)
        junc[:, cols] = np.stack([rng.multinomial(int(n), q) for n, q in zip(N_e[:, a], p)])
        clus[:, cols] = N_e[:, [a]]
    return dict(junc=junc, clus=clus, event_id=event_id, K=K)


def main():
    sim = simulate()
    print("posterior WIDTH of psi -- the mechanism the DM is meant to fix")
    print(f"{'config':<34} {'median psi sd':>14} {'vs BB+beta':>11}")
    base = None
    for lik, par in (("betabinomial", "beta"), ("betabinomial", "simplex"),
                     ("dirichletmultinomial", "simplex")):
        pyro.clear_param_store()
        torch.manual_seed(0)
        m = LeafletFAMinibatch(sim["junc"], sim["clus"], K=sim["K"], device="cpu",
                               likelihood=lik, psi_parameterization=par,
                               event_id=sim["event_id"])
        m.fit(num_steps=500, subsample_size=256, lr=0.05, gamma=0.5, seed=0, verbose=False)
        ps = m.sample_psi(300, seed=0).cpu().numpy()          # [S, K, J] on the psi scale
        sd = float(np.median(ps.std(0)))
        if base is None:
            base = sd
        print(f"{lik + ' + ' + par:<34} {sd:>14.5f} {sd / base:>10.2f}x")


if __name__ == "__main__":
    main()
