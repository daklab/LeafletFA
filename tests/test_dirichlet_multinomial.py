#!/usr/bin/env python
"""Pin the segment DM log-prob against pyro's own DirichletMultinomial.

The whole argument for the DM rests on this function being correct, so it is checked against
an independent implementation on ragged events, on degenerate events, and on the zero-coverage
case that the model relies on contributing nothing.

Run: python tests/test_dirichlet_multinomial.py
"""

import sys

import numpy as np
import torch
import pyro.distributions as dist

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.dirichlet_multinomial import (  # noqa: E402
    build_atse_index, dm_log_prob, check_A_e_is_conc, denominator_consistency)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")


def reference_log_prob(y_rows, alpha_rows):
    """Independent reference: pyro DirichletMultinomial, one event at a time, minus the
    multinomial coefficient (which dm_log_prob deliberately omits)."""
    total = 0.0
    for y_e, a_e in zip(y_rows, alpha_rows):
        y_e = torch.as_tensor(y_e, dtype=torch.float)
        a_e = torch.as_tensor(a_e, dtype=torch.float)
        N = float(y_e.sum())
        lp = dist.DirichletMultinomial(concentration=a_e, total_count=int(N)).log_prob(y_e)
        coef = torch.lgamma(torch.tensor(N + 1)) - torch.lgamma(y_e + 1).sum()
        total += float(lp - coef)
    return total


print("1. index construction on ragged events")
# events: e0 = {j0,j1}, e1 = {j2,j3,j4}, e2 = {j5}
event_id = np.array(["e0", "e0", "e1", "e1", "e1", "e2"])
atse_idx, rep_idx, sizes = build_atse_index(event_id)
check("atse_idx maps each junction to its event", atse_idx.tolist() == [0, 0, 1, 1, 1, 2],
      str(atse_idx.tolist()))
check("sizes counts junctions per event", sizes.tolist() == [2, 3, 1], str(sizes.tolist()))
check("rep_idx is the first junction of each event", rep_idx.tolist() == [0, 2, 5],
      str(rep_idx.tolist()))

print("\n1b. index is order-independent (labels not pre-sorted)")
shuffled = np.array(["z", "a", "z", "m", "a", "m"])
ai, ri, sz = build_atse_index(shuffled)
groups = {}
for j, e in enumerate(ai.tolist()):
    groups.setdefault(e, []).append(j)
check("junctions with the same label land in the same event",
      sorted(map(sorted, groups.values())) == [[0, 2], [1, 4], [3, 5]], str(groups))
check("rep_idx points into its own event",
      all(ai[int(r)] == e for e, r in enumerate(ri)), str(ri.tolist()))
check("sizes sums to J", int(sz.sum()) == 6)

print("\n2. matches pyro DirichletMultinomial, ragged events, one cell")
y = torch.tensor([[7., 3., 5., 2., 1., 4.]])
clus = torch.tensor([[10., 10., 8., 8., 8., 4.]])       # ATSE totals broadcast per event
mu = torch.tensor([[0.6, 0.4, 0.5, 0.3, 0.2, 1.0]])     # simplex-valid within each event
conc = torch.tensor(9.0)
got = float(dm_log_prob(y, mu, conc, atse_idx, 3))
want = reference_log_prob([[7., 3.], [5., 2., 1.], [4.]],
                          [[5.4, 3.6], [4.5, 2.7, 1.8], [9.0]])
check("segment log-prob == pyro reference", abs(got - want) < 1e-3, f"{got:.6f} vs {want:.6f}")

print("\n3. matches pyro with UNCONSTRAINED psi (A_e != conc)")
mu_free = torch.tensor([[0.9, 0.4, 0.5, 0.3, 0.9, 0.2]])   # deliberately does not sum to 1
got = float(dm_log_prob(y, mu_free, conc, atse_idx, 3))
a = (conc * mu_free)[0].tolist()
want = reference_log_prob([[7., 3.], [5., 2., 1.], [4.]],
                          [a[0:2], a[2:5], a[5:6]])
check("segment log-prob == pyro reference, unconstrained", abs(got - want) < 1e-3,
      f"{got:.6f} vs {want:.6f}")

print("\n4. multiple cells at once")
yb = torch.tensor([[7., 3., 5., 2., 1., 4.], [1., 1., 0., 1., 1., 2.]])
clusb = torch.tensor([[10., 10., 8., 8., 8., 4.], [2., 2., 2., 2., 2., 2.]])
mub = mu.repeat(2, 1)
gotb = dm_log_prob(yb, mub, conc, atse_idx, 3)
w0 = reference_log_prob([[7., 3.], [5., 2., 1.], [4.]],
                        [[5.4, 3.6], [4.5, 2.7, 1.8], [9.0]])
w1 = reference_log_prob([[1., 1.], [0., 1., 1.], [2.]],
                        [[5.4, 3.6], [4.5, 2.7, 1.8], [9.0]])
check("batched rows match per-row reference",
      abs(float(gotb[0]) - w0) < 1e-3 and abs(float(gotb[1]) - w1) < 1e-3,
      f"{gotb.tolist()} vs [{w0:.4f}, {w1:.4f}]")

print("\n5. zero-coverage events contribute exactly zero (no mask needed)")
y_z = torch.tensor([[7., 3., 0., 0., 0., 4.]])           # event 1 has no reads in this cell
got_z = float(dm_log_prob(y_z, mu, conc, atse_idx, 3))
want_z = reference_log_prob([[7., 3.], [4.]], [[5.4, 3.6], [9.0]])
check("uncovered event drops out", abs(got_z - want_z) < 1e-4, f"{got_z:.6f} vs {want_z:.6f}")

print("\n6. a fully uncovered cell has log-prob 0")
got0 = float(dm_log_prob(torch.zeros(1, 6), mu, conc, atse_idx, 3))
check("all-zero cell contributes nothing", abs(got0) < 1e-5, f"{got0:.3e}")

print("\n7. effective n: the DM counts an event ONCE, the BetaBinomial sum counts it m times")
# same data, both likelihoods, on a single 4-junction event
y1 = torch.tensor([[5., 3., 1., 1.]])
N = 10.0
clus1 = torch.full((1, 4), N)
mu1 = torch.tensor([[0.5, 0.3, 0.1, 0.1]])
ai1, ri1, _ = build_atse_index(np.array(["e", "e", "e", "e"]))
lp_dm = float(dm_log_prob(y1, mu1, conc, ai1, 1))
bb = dist.BetaBinomial(concentration1=mu1 * conc, concentration0=(1 - mu1) * conc,
                       total_count=clus1)
lp_bb = float(bb.log_prob(y1).sum())
check("DM and BetaBinomial-sum disagree, as they must", abs(lp_dm - lp_bb) > 0.5,
      f"DM {lp_dm:.4f} vs BB-sum {lp_bb:.4f}")

# curvature: perturb psi_0 and compare how sharply each responds
def curvature(fn, h=1e-3):
    base = torch.tensor([0.5, 0.3, 0.1, 0.1])
    out = []
    for d in (-h, 0.0, h):
        p = base.clone(); p[0] += d; p[1] -= d
        out.append(fn(p[None]))
    return (out[0] - 2 * out[1] + out[2]) / h ** 2

c_dm = curvature(lambda m: float(dm_log_prob(y1, m, conc, ai1, 1)))
c_bb = curvature(lambda m: float(dist.BetaBinomial(
    concentration1=m * conc, concentration0=(1 - m) * conc, total_count=clus1)
    .log_prob(y1).sum()))
check("BetaBinomial-sum is the sharper (over-confident) likelihood", abs(c_bb) > abs(c_dm),
      f"|d2 BB| {abs(c_bb):.1f} > |d2 DM| {abs(c_dm):.1f}, ratio {abs(c_bb / c_dm):.2f}")

print("\n8. A_e diagnostic separates constrained from unconstrained psi")
mx, md = check_A_e_is_conc(mu, conc, atse_idx, 3)
check("simplex psi gives A_e == conc", mx < 1e-5, f"max rel dev {mx:.2e}")
mx2, md2 = check_A_e_is_conc(mu_free, conc, atse_idx, 3)
check("unconstrained psi is flagged", mx2 > 0.1, f"max rel dev {mx2:.2e}")

print("\n9. gradients flow to mu and conc")
mu_g = mu.clone().requires_grad_(True)
conc_g = torch.tensor(9.0, requires_grad=True)
dm_log_prob(y, mu_g, conc_g, atse_idx, 3).sum().backward()
check("d/dmu is finite and non-zero", mu_g.grad is not None
      and bool(torch.isfinite(mu_g.grad).all()) and float(mu_g.grad.abs().sum()) > 0)
check("d/dconc is finite and non-zero", conc_g.grad is not None
      and bool(torch.isfinite(conc_g.grad)) and abs(float(conc_g.grad)) > 0,
      f"{float(conc_g.grad):.4f}")

print("\n10. denominator_consistency separates the real atlas from the simulations")
# atlas-like: clus IS the segment sum, so the gap is zero
y_ok = torch.tensor([[7., 3., 5., 2., 1., 4.]])
clus_ok = torch.tensor([[10., 10., 8., 8., 8., 4.]])
mxd, frac = denominator_consistency(y_ok, clus_ok, atse_idx, 3, rep_idx)
check("consistent denominators report a zero gap", mxd < 1e-5 and frac == 0.0,
      f"max {mxd:.3g}, frac {frac:.3g}")
# simulation-like: junctions drawn independently, so they overshoot the stored total
y_sim = torch.tensor([[1., 1., 5., 2., 1., 4.]])
clus_sim = torch.tensor([[1., 1., 8., 8., 8., 4.]])      # event 0 stores 1 but y sums to 2
mxd2, frac2 = denominator_consistency(y_sim, clus_sim, atse_idx, 3, rep_idx)
check("independent-Binomial simulation is flagged", mxd2 > 0.5 and frac2 > 0,
      f"max {mxd2:.3g}, frac {frac2:.3g}")
# and the DM is unaffected, because it derives N_e from y
lp_sim = float(dm_log_prob(y_sim, mu, conc, atse_idx, 3))
want_sim = reference_log_prob([[1., 1.], [5., 2., 1.], [4.]],
                              [[5.4, 3.6], [4.5, 2.7, 1.8], [9.0]])
check("dm_log_prob still correct when the stored total disagrees",
      abs(lp_sim - want_sim) < 1e-3, f"{lp_sim:.6f} vs {want_sim:.6f}")

print("\n11. pyro.factor gets the plate's N/B scaling, same as pyro.sample")
# If the DM factor were NOT scaled, the likelihood would be down-weighted by N/B relative to
# the prior at every step -- catastrophic and silent. Verified via the ELBO SVI actually sees.
import pyro                                                          # noqa: E402
from pyro import poutine                                             # noqa: E402
from pyro.infer import Trace_ELBO                                    # noqa: E402

NCELL = 1000


def _mk(kind, B):
    def model():
        with pyro.plate("cells", NCELL, subsample_size=B) as idx:
            if kind == "factor":
                pyro.factor("obs", torch.ones(len(idx)))
            else:
                pyro.sample("obs", dist.Normal(torch.zeros(len(idx)), 1.0),
                            obs=torch.zeros(len(idx)))
    return model


scales = []
for B in (100, 500, 1000):
    tr = poutine.trace(_mk("factor", B)).get_trace()
    tr.compute_log_prob()
    scales.append((B, float(tr.nodes["obs"]["scale"]), float(tr.nodes["obs"]["log_prob"].sum())))
check("plate scale is N/B at every batch size",
      all(abs(sc - NCELL / B) < 1e-6 for B, sc, _ in scales), str([(b, s) for b, s, _ in scales]))
check("scaled log-prob is batch-size invariant",
      max(v for _, _, v in scales) - min(v for _, _, v in scales) < 1e-3,
      str([round(v, 2) for _, _, v in scales]))

losses = {}
for kind in ("factor", "sample"):
    losses[kind] = [Trace_ELBO(num_particles=20).loss(_mk(kind, B), lambda: None)
                    for B in (100, 1000)]
check("ELBO for a factor is batch-size invariant",
      abs(losses["factor"][0] - losses["factor"][1]) < 1e-3, str(losses["factor"]))
check("factor and sample branches scale identically",
      abs(abs(losses["factor"][0]) - NCELL) < 1e-3
      and abs(losses["sample"][0] - losses["sample"][1]) < 1e-3,
      f"factor {losses['factor'][0]:.2f}, sample {losses['sample'][0]:.2f}")

print(f"\n{'=' * 62}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
