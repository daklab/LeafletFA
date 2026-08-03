#!/usr/bin/env python
"""Why the DM matters: it is not the log-prob VALUE, it is the curvature.

Run it: python 02_posterior_width_vs_event_size.py

Posterior width is set by the curvature of the log-likelihood at its peak, not by its height.
Counting one event m times makes the log-likelihood roughly m times steeper, so the posterior
comes out too narrow -- which is precisely the over-confidence that makes the reported FDR
3-4x anti-conservative.

This sweeps psi_0 for a single ATSE and reports the Laplace standard deviation under
  (a) the current sum of independent BetaBinomials,
  (b) the Dirichlet-multinomial, which is correct for the same data,
  (c) A35's (m-1)/m down-weighting, the scalar approximation.

The headline: the gap grows with event size, and (m-1)/m barely closes it for large events.
That is why A35's weighting calibrated effect 0.25 but stalled at effect 0.50, and it predicts
where the DM should win.
"""

import torch
import pyro.distributions as dist


def curves(y, conc=8.0):
    """Log-likelihood of psi_0 over a grid, under both likelihoods, for one ATSE."""
    m = len(y)
    N_e = float(sum(y))
    grid = torch.linspace(0.05, 0.95, 1801)
    rest = torch.tensor(y[1:], dtype=torch.float)
    rest = rest / rest.sum()                     # hold the other junctions' relative usage fixed
    yv = torch.tensor(y, dtype=torch.float)
    bb_sum, dm_lp = [], []
    for p0 in grid:
        psi = torch.cat([p0[None], (1 - p0) * rest])
        bb = dist.BetaBinomial(concentration1=psi * conc, concentration0=(1 - psi) * conc,
                               total_count=torch.full((m,), N_e))
        bb_sum.append(bb.log_prob(yv).sum())
        dm_lp.append(dist.DirichletMultinomial(concentration=psi * conc,
                                               total_count=int(N_e)).log_prob(yv))
    return grid, torch.stack(bb_sum), torch.stack(dm_lp), m


def width(grid, lp):
    """Laplace sd from the second derivative at the peak."""
    i = int(lp.argmax())
    i = min(max(i, 1), len(lp) - 2)
    h = float(grid[1] - grid[0])
    d2 = float((lp[i + 1] - 2 * lp[i] + lp[i - 1]) / h ** 2)
    return (-1.0 / d2) ** 0.5 if d2 < 0 else float("nan")


print(f"{'ATSE (junction counts)':<28} {'m':>2} {'sd BB-sum':>10} {'sd DM':>8} "
      f"{'DM/BB':>7} {'sqrt(m)':>8} {'sd (m-1)/m':>11}")
print("-" * 82)
for y in ([7, 3], [7, 2, 1], [5, 3, 1, 1], [4, 2, 2, 1, 1], [10, 4, 3, 2, 1, 1, 1, 1]):
    grid, bb, dm, m = curves(y)
    s_bb, s_dm = width(grid, bb), width(grid, dm)
    s_w = width(grid, bb * (m - 1) / m)          # the A35 approximation
    print(f"{str(y):<28} {m:>2} {s_bb:>10.4f} {s_dm:>8.4f} {s_dm / s_bb:>7.2f} "
          f"{m ** 0.5:>8.2f} {s_w:>11.4f}")

print("""
Reading the table
-----------------
'sd BB-sum' is how wide the CURRENT likelihood believes psi_0 to be; 'sd DM' is the correct
width for the same counts. The ratio grows with event size -- about 1.4x at m = 2, about 3x at
m = 8 -- so the current model is most over-confident exactly on the large ATSEs, which are also
where composition validity collapses (47% valid at 6+ junctions) and where the strong effects
live. It tracks sqrt(m) loosely; the exact ratio also depends on the count pattern, so treat
sqrt(m) as the intuition and not as a formula.

The last column is A35's weighting. At m = 2 it lands near the DM. At m = 8 it moves the width
from 0.050 to 0.053 where the DM says 0.154 -- almost nothing. A single scalar cannot correct a
per-event quantity, which is the whole argument for doing it properly.""")
