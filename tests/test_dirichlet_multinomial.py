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

print("\n12. the DM must not be SLOWER than the BetaBinomial it replaces")
# The DM only stays cheap because atse_idx is a flat length-J integer array, so grouping is one
# index_add pass over a contiguous tensor. A rewrite that loops over events, or pads to max(m),
# would still be CORRECT and would silently cost ~10x. Measured at atlas shape the DM is ~0.2x
# the BetaBinomial; assert only "not slower", which is loose enough not to flake on a busy node.
import time as _time                                                  # noqa: E402

_sizes = np.array([2] * 400 + [3] * 300 + [4] * 120 + [8] * 80)
_J = int(_sizes.sum())
_ai, _ri, _sz = build_atse_index(np.repeat(np.arange(len(_sizes)), _sizes))
_na, _B = len(_sz), 256
_rng = np.random.default_rng(0)
_y = torch.from_numpy(_rng.poisson(0.5, size=(_B, _J)).astype(np.float32))
_mu = torch.rand(_B, _J) * 0.5 + 0.25
_c = torch.tensor(20.0)
_cl = torch.from_numpy(_rng.poisson(5, size=(_B, _J)).astype(np.float32)) + 1


def _time_it(fn, n=5):
    fn()
    t0 = _time.time()
    for _ in range(n):
        fn()
    return (_time.time() - t0) / n


_t_dm = _time_it(lambda: dm_log_prob(_y, _mu, _c, _ai, _na))
_bb = dist.BetaBinomial(concentration1=_mu * _c, concentration0=(1 - _mu) * _c, total_count=_cl)
_t_bb = _time_it(lambda: _bb.log_prob(_y).sum(-1))
check("dm_log_prob is not slower than BetaBinomial", _t_dm <= _t_bb,
      f"DM {_t_dm*1000:.1f} ms vs BB {_t_bb*1000:.1f} ms (ratio {_t_dm/_t_bb:.2f})")

print("\n13. the alpha floor bounds the gradient that broke every atlas fit")
# Atlas fits drove psi into the clip (1e-4) and learned conc 0.81-1.9, putting alpha near 1e-4
# where d/da [lgamma(y+a) - lgamma(a)] ~ 1/a reaches 5,000-12,000 per entry. The previous floor of
# 1e-6 sat 190x below that and never fired, which is why it never prevented anything. This pins
# both halves: the floor MUST bite in the atlas regime and MUST NOT perturb the ordinary one.
_fmu = torch.tensor([[0.6, 0.3999, 1e-4]])
_fy = torch.tensor([[20., 10., 1.]])          # a read on a junction the model called ~0
_fai, _, _ = build_atse_index(np.array(["e", "e", "e"]))


def _maxgrad(floor, conc=1.904):
    a = (conc * _fmu).clamp_min(floor).clone().requires_grad_(True)
    per = (torch.lgamma(_fy + a) - torch.lgamma(a)).sum(-1)
    A, N = a.sum(-1), _fy.sum(-1)
    (per + torch.lgamma(A) - torch.lgamma(N + A)).sum().backward()
    return float(a.grad.abs().max())


_g_lo, _g_hi = _maxgrad(1e-6), _maxgrad(1e-2)
check("the floor bounds the atlas-regime gradient", _g_hi < 200 < 1000 < _g_lo,
      f"eps=1e-6 -> {_g_lo:.0f}, eps=1e-2 -> {_g_hi:.0f}")

_oy = torch.tensor([[30., 12., 8.], [5., 40., 2.]])
_omu = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
_olo = dm_log_prob(_oy, _omu, torch.tensor(50.0), _fai, 1, eps=1e-6)
_ohi = dm_log_prob(_oy, _omu, torch.tensor(50.0), _fai, 1, eps=1e-2)
check("the floor does not perturb the ordinary regime", torch.allclose(_olo, _ohi, atol=1e-6),
      f"eps=1e-6 {float(_olo.sum()):.6f} vs eps=1e-2 {float(_ohi.sum()):.6f}")

print("\n14. the NaN-guard rollback must restore CONSTRAINED parameters too")
# ParamStoreDict.__getitem__ returns transform_to(constraint)(unconstrained), built fresh on every
# access. Snapshotting .items() and copy_()-ing back therefore writes into a temporary and is
# silently discarded -- for constrained parameters only. AutoNormal holds every scale under
# constraints.positive, so the old rollback restored the guide's locs and left its scales at
# whatever value diverged: atlas fits were saved with psi_loc finite and psi_scale all-NaN while
# the guard reported a successful restore. Identity must also survive, or the optimiser keeps
# updating tensors nobody reads.
import pyro as _pyro
from pyro.distributions import constraints as _constraints

_pyro.clear_param_store()
_st = _pyro.get_param_store()
_pyro.param("u_loc", torch.tensor([1.0, 2.0]))
_pyro.param("c_scale", torch.tensor([1.0, 2.0]), constraint=_constraints.positive)
_snap = {"params": {k: v.detach().cpu().clone() for k, v in _st.get_state()["params"].items()},
         "constraints": dict(_st.get_state()["constraints"])}
_ids = {k: id(v) for k, v in _st._params.items()}
for _k in list(_st._params):
    _st._params[_k].data.fill_(float("nan"))

# the same in-place restore LeafletFAMinibatch._restore_snapshot performs
with torch.no_grad():
    for _k, _v in _snap["params"].items():
        _p = _st._params.get(_k)
        if _p is not None and _p.shape == _v.shape:
            _p.data.copy_(_v.to(_p.device))

check("rollback restores the UNCONSTRAINED parameter",
      bool(torch.allclose(_st["u_loc"].detach(), torch.tensor([1.0, 2.0]))),
      f"u_loc -> {_st['u_loc'].detach().tolist()}")
check("rollback restores the CONSTRAINED parameter",
      bool(torch.allclose(_st["c_scale"].detach(), torch.tensor([1.0, 2.0]))),
      f"c_scale -> {_st['c_scale'].detach().tolist()}")
check("tensor identity survives, so the optimiser keeps its references",
      _ids == {k: id(v) for k, v in _st._params.items()})
_pyro.clear_param_store()

print("\n15. a NaN must not end the run: restore, clear optimiser state, continue")
# Restoring parameters alone is not enough. Adam keeps exp_avg / exp_avg_sq per parameter; a
# non-finite gradient poisons those buffers, and the next update writes NaN straight back into the
# restored parameters. Measured on the atlas before the state clear: the guard fired at 2016, 2018,
# 2020, 2022, 2024, 2026, 2028 -- every second step, exhausting all six restarts in twelve steps
# and ending the fit at 12% of its budget. One injected hit must now cost one restart, not all six.
import pyro as _p2
import leafletfa.minibatch as _MB
from leafletfa.minibatch import LeafletFAMinibatch as _LFA

_rng = np.random.default_rng(0)
_N, _K = 200, 3
_ev2 = np.array([f"e{i//3}" for i in range(24)])
_na2 = len(np.unique(_ev2))
_Ne = _rng.integers(5, 50, size=(_N, _na2)).astype(np.float32)
_clus2 = _Ne[:, np.repeat(np.arange(_na2), 3)]
_junc2 = np.zeros((_N, len(_ev2)), np.float32)
for _e in range(_na2):
    _pp = _rng.dirichlet([2, 2, 2])
    for _i in range(_N):
        _junc2[_i, 3*_e:3*_e+3] = _rng.multinomial(int(_Ne[_i, _e]), _pp)

_real = _MB.dm_log_prob
_cnt = {"n": 0}


def _flaky(y, mu, conc, atse_idx, n_atse_, eps=1e-2, event_weight=None):
    _cnt["n"] += 1
    out = _real(y, mu, conc, atse_idx, n_atse_, eps=eps, event_weight=event_weight)
    return out * torch.tensor(float("nan")) if _cnt["n"] == 120 else out


_MB.dm_log_prob = _flaky
try:
    _p2.clear_param_store()
    _m2 = _LFA(_junc2, _clus2, K=_K, device="cpu", likelihood="dirichletmultinomial",
               psi_parameterization="simplex", event_id=_ev2, conc_per_atse=True)
    _L2 = _m2.fit(num_steps=300, subsample_size=64, lr=0.05, seed=0, verbose=False,
                  early_stop=False)
finally:
    _MB.dm_log_prob = _real

check("one non-finite step costs exactly one restart", _m2.n_nan_restarts == 1,
      f"restarts={_m2.n_nan_restarts} at steps {_m2.nan_steps}")
check("the run continues to the full step budget", len(_L2) >= 299 and np.isfinite(_L2[-1]),
      f"{len(_L2)}/300 steps, final loss {_L2[-1]:.4e}")
_lc, _sc2 = _m2.psi_loc_scale()
check("guide loc AND scale are finite afterwards",
      bool(np.isfinite(_lc).all() and np.isfinite(_sc2).all()))
_p2.clear_param_store()

print(f"\n{'=' * 62}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
