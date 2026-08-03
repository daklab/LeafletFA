#!/usr/bin/env python
"""Is the Dirichlet-multinomial affordable at atlas scale? Yes -- it is cheaper.

Run: python 05_cost_at_atlas_scale.py

The worry before measuring was that the DM adds two segment reductions over [B, J] per step and
would slow an already-expensive atlas fit. It does add them, but it also REMOVES work: pyro's
BetaBinomial log_prob evaluates several lgamma terms over [B, J] (including the binomial
coefficient), while the DM needs only lgamma(y + a) and lgamma(a) plus two cheap index_add
passes. The segment reductions are linear scatters, far cheaper than an elementwise lgamma.

Measured on an atlas-shaped problem (77,114 junctions, 24,475 ATSEs of size 2-8), forward pass,
4 CPU threads:

    B =  512    DM   855 ms    BetaBinomial  2682 ms    DM/BB 0.32x
    B = 2048    DM  3975 ms    BetaBinomial 12064 ms    DM/BB 0.33x

So the DM is about 3x faster per likelihood evaluation. Cost is not an argument against
retraining the atlas with it.
"""

import sys
import time

import numpy as np
import torch
import pyro.distributions as dist

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.dirichlet_multinomial import build_atse_index, dm_log_prob  # noqa: E402

torch.set_num_threads(4)

rng = np.random.default_rng(0)
# ATSE size distribution close to the real atlas
sizes = np.array([2] * 10677 + [3] * 7600 + [4] * 2381 + [5] * 1500
                 + [6] * 900 + [7] * 800 + [8] * 617)
J = int(sizes.sum())
event_id = np.repeat(np.arange(len(sizes)), sizes)
atse_idx, rep_idx, sz = build_atse_index(event_id)
n_atse = len(sz)
print(f"atlas-like: J={J:,}, ATSEs={n_atse:,}, sizes {sizes.min()}-{sizes.max()}\n")

for B in (512, 2048):
    y = torch.from_numpy(rng.poisson(0.5, size=(B, J)).astype(np.float32))
    mu = torch.rand(B, J) * 0.5 + 0.25
    conc = torch.tensor(20.0)
    clus = torch.from_numpy(rng.poisson(5, size=(B, J)).astype(np.float32)) + 1

    dm_log_prob(y, mu, conc, atse_idx, n_atse)                      # warm up
    t = time.time()
    for _ in range(3):
        dm_log_prob(y, mu, conc, atse_idx, n_atse)
    t_dm = (time.time() - t) / 3

    bb = dist.BetaBinomial(concentration1=mu * conc, concentration0=(1 - mu) * conc,
                           total_count=clus)
    bb.log_prob(y)                                                  # warm up
    t = time.time()
    for _ in range(3):
        bb.log_prob(y).sum(-1)
    t_bb = (time.time() - t) / 3

    print(f"  B={B:5d}  DM {t_dm*1000:7.1f} ms   BetaBinomial {t_bb*1000:7.1f} ms   "
          f"DM/BB {t_dm/t_bb:.2f}x")

print("\nForward pass only; both carry roughly the same backward multiplier.")
