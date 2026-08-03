#!/usr/bin/env python
"""Smoke test the 2x2 of {likelihood} x {psi parameterisation} on a compositional simulation.

Run: python 03_factorial_smoke.py [--steps 600] [--seeds 0 1 2]

The simulation draws each ATSE's junction counts as ONE multinomial from a true composition, so
sum_j y_j == N_e exactly and the true psi sums to 1 within every event. That is the regime the
Dirichlet-multinomial and the simplex are designed for, which makes it the right place to check
that the implementation recovers what it should before anything runs at scale.

What each cell should do, stated before the numbers:
  betabinomial + beta     the current model. Recovers psi; within-ATSE sums land near 1 only
                          because this sim is small and well specified.
  betabinomial + simplex  sums exactly 1 by construction. Recovery should not degrade.
  dirichletmultinomial + beta      the DM depends on alpha only through conc * psi, so the
                          within-event SCALE of psi is unidentified. Sums are expected to drift
                          away from 1. This cell exists to show the DM needs the simplex.
  dirichletmultinomial + simplex   the intended model: sums exactly 1, recovery intact.
"""

import argparse
import sys

import numpy as np
import torch
import pyro
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.minibatch import LeafletFAMinibatch  # noqa: E402


def simulate(N=1500, A=60, m=3, K=4, seed=0, mean_cov=15):
    """Counts drawn as one multinomial per (cell, ATSE) from a true composition."""
    rng = np.random.default_rng(seed)
    J = A * m
    event_id = np.repeat(np.arange(A), m)
    psi = rng.dirichlet(np.ones(m), size=(K, A)).reshape(K, J)     # sums to 1 within each event
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
    return dict(junc=junc, clus=clus, psi=psi, phi=phi, event_id=event_id, K=K, A=A, m=m)


def score(est, sim):
    """Matched psi recovery and the within-ATSE sum, the two things we care about."""
    K, A, m = sim["K"], sim["A"], sim["m"]
    C = np.corrcoef(est, sim["psi"])[:K, K:]
    r, c = linear_sum_assignment(-C)
    S = est.reshape(K, A, m).sum(2)
    return (float(np.mean([C[i, j] for i, j in zip(r, c)])),
            float(np.median(S)), float(((S >= 0.9) & (S <= 1.1)).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    sim = simulate()
    gap = np.abs(sim["junc"].reshape(-1, sim["A"], sim["m"]).sum(2)
                 - sim["clus"][:, ::sim["m"]]).max()
    print(f"sim: {sim['junc'].shape[0]} cells, {sim['A']} ATSEs x {sim['m']} junctions, K={sim['K']}")
    print(f"     sum_j y_j == N_e exactly: max gap {gap:.0f}\n")

    combos = [(lik, par) for lik in ("betabinomial", "dirichletmultinomial")
              for par in ("beta", "simplex")]
    rows = []
    for lik, par in combos:
        rs, meds, valids = [], [], []
        for seed in args.seeds:
            pyro.clear_param_store()
            torch.manual_seed(seed)
            m = LeafletFAMinibatch(sim["junc"], sim["clus"], K=sim["K"], device="cpu",
                                   likelihood=lik, psi_parameterization=par,
                                   event_id=sim["event_id"])
            m.fit(num_steps=args.steps, subsample_size=args.batch, lr=0.05, gamma=0.5,
                  seed=seed, verbose=False)
            est = m.psi_mean()
            est = est.cpu().numpy() if hasattr(est, "cpu") else np.asarray(est)
            r, med, valid = score(est, sim)
            rs.append(r); meds.append(med); valids.append(valid)
        rows.append((lik, par, np.mean(rs), np.std(rs), np.mean(meds), 100 * np.mean(valids)))
        print(f"  {lik:>20} + {par:<8}  r={np.mean(rs):.4f}+-{np.std(rs):.4f}  "
              f"sum={np.mean(meds):.4f}  valid={100*np.mean(valids):.1f}%", flush=True)

    print(f"\n{'likelihood':>20}  {'psi':<8} {'recovery r':>12} {'seed sd':>9} "
          f"{'median sum':>11} {'valid %':>8}")
    print("-" * 74)
    for lik, par, r, sd, med, valid in rows:
        print(f"{lik:>20}  {par:<8} {r:>12.4f} {sd:>9.4f} {med:>11.4f} {valid:>7.1f}%")


if __name__ == "__main__":
    main()
