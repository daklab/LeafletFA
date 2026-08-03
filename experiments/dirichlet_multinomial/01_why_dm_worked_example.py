#!/usr/bin/env python
"""Worked example: what the current likelihood does to one ATSE, vs what the DM does.

Run it: python 01_why_dm_worked_example.py

The model today (leafletfa/minibatch.py:226-231) puts an independent BetaBinomial on every
junction and sums them with `.to_event(1)`. Every junction of an ATSE carries the SAME
denominator -- the ATSE total -- so one event's worth of evidence is counted once per junction.
For an event with m junctions the likelihood behaves as if it had seen m observations.

The Dirichlet-multinomial is the distribution that describes the event correctly: one draw of
a count VECTOR from one total. This script shows the two side by side on a single ATSE, then
shows that the segment formula used in `leafletfa/dirichlet_multinomial.py` reproduces pyro's
`dist.DirichletMultinomial` exactly.
"""

import torch
import numpy as np
import pyro.distributions as dist

torch.set_printoptions(precision=4, sci_mode=False)

# ---- one ATSE with 3 junctions, one cell --------------------------------------------------
y = torch.tensor([7., 2., 1.])       # junction counts in this cell
N_e = y.sum()                        # ATSE total = 10 -- this is cell_by_cluster_matrix
psi = torch.tensor([0.6, 0.3, 0.1])  # a program's usage; sums to 1 (the simplex constraint)
conc = torch.tensor(8.0)             # overdispersion

print(f"ATSE: y = {[int(v) for v in y]},  N_e = {N_e:.0f},  "
      f"psi = {[round(float(v), 2) for v in psi]},  conc = {conc:.0f}\n")

# ---- A. WHAT THE MODEL DOES TODAY: m independent BetaBinomials ---------------------------
bb = dist.BetaBinomial(concentration1=psi * conc,
                       concentration0=(1 - psi) * conc,
                       total_count=N_e.expand(3))
lp_each = bb.log_prob(y)
print("A. current: one BetaBinomial per junction, summed by .to_event(1)")
for j in range(3):
    print(f"   junction {j}: y={y[j]:.0f} of N_e={N_e:.0f}  ->  log p = {lp_each[j]:+.4f}")
print(f"   sum                                     = {lp_each.sum():+.4f}")
print(f"   >> N_e = {N_e:.0f} was the denominator 3 TIMES. One event, entered as m = 3.\n")

# ---- B. THE DM: one term for the whole event ---------------------------------------------
a = psi * conc
lp_dm = dist.DirichletMultinomial(concentration=a, total_count=int(N_e)).log_prob(y)
print("B. Dirichlet-multinomial: ONE term for the event")
print(f"   log p = {lp_dm:+.4f}")
print("   >> N_e enters once. That is the effective-n correction, exactly.\n")

# ---- C. THE SEGMENT FORMULA REPRODUCES B --------------------------------------------------
A_e = a.sum()                        # == conc, because psi sums to 1 within the event
per_junction = (torch.lgamma(y + a) - torch.lgamma(a)).sum()
per_event = torch.lgamma(A_e) - torch.lgamma(N_e + A_e)
mult_coef = torch.lgamma(N_e + 1) - torch.lgamma(y + 1).sum()   # constant in the parameters

print("C. the segment formula (what leafletfa/dirichlet_multinomial.py computes)")
print(f"   A_e = sum_j a_j                       = {A_e:.4f}  (== conc, since psi sums to 1)")
print(f"   sum_j [lgamma(y_j+a_j) - lgamma(a_j)] = {per_junction:+.4f}  <- plain sum, no grouping")
print(f"   lgamma(A_e) - lgamma(N_e+A_e)         = {per_event:+.4f}  <- ONE per event")
print(f"   multinomial coefficient               = {mult_coef:+.4f}  <- constant, no gradient")
print(f"   total                                 = {per_junction + per_event + mult_coef:+.4f}")
print(f"   pyro DirichletMultinomial             = {lp_dm:+.4f}")
print(f"   match: {torch.allclose(per_junction + per_event + mult_coef, lp_dm, atol=1e-4)}\n")

print("""Why A_e needs no segment sum
---------------------------
a_j = conc * psi_j, and the simplex constraint makes sum_{j in e} psi_j = 1 exactly. So
A_e = conc * 1 = conc for EVERY event -- one scalar, not a per-event quantity. This is the
reason the DM is cheap here and the reason it requires the simplex: without sum psi = 1 the
event concentration varies per event and per program, and would have to be gathered.

Why a pyro.factor and not dist.DirichletMultinomial
---------------------------------------------------
Events are ragged: 2 to 11+ junctions. The distribution needs a fixed last dimension, so
using it means gathering counts into a padded [B, n_atse, max_m] tensor and masking the pad,
every step, on top of the [B, J] tensors already being moved. The segment form consumes the
[B, J] layout as-is and needs one precomputed index of length n_atse.""")
