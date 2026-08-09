"""Dirichlet-multinomial likelihood over ATSEs, evaluated with segment operations.

Why this exists
---------------
The model's default likelihood puts an independent BetaBinomial on every junction and sums
them (`minibatch.py`, the `obs` site). Every junction of an ATSE carries the SAME denominator,
because `cell_by_cluster_matrix[c, j]` is the ATSE total broadcast to each of its junctions.
So an event with m junctions enters the likelihood as if it were m independent observations.
The log-likelihood is roughly m times too steep, the posterior comes out about sqrt(m) times
too narrow, and the reported FDR is anti-conservative as a result.

The Dirichlet-multinomial is the distribution that describes an ATSE correctly: one draw of a
count VECTOR from one total. See `experiments/dirichlet_multinomial/` for the worked example
and for the measured posterior widths.

Why a factor and not `dist.DirichletMultinomial`
------------------------------------------------
Events are ragged -- 2 to 11+ junctions -- and the distribution needs a fixed trailing
dimension. Using it means gathering counts into a padded [B, n_atse, max_m] tensor and masking
the padding on every step, on top of the [B, J] tensors already being moved. Evaluating the
log-probability directly consumes the [B, J] layout as-is:

    log p(y_e | a_e) = lgamma(A_e) - lgamma(N_e + A_e)
                       + sum_{j in e} [lgamma(y_j + a_j) - lgamma(a_j)]

with a_j = conc * mu_j, A_e = sum_{j in e} a_j and N_e = sum_{j in e} y_j. The junction sum
needs no grouping at all, since each junction belongs to exactly one event; only A_e and N_e
are per-event, and both are one `index_add` over the [B, J] tensors already in hand.

N_e is DERIVED, never read from cell_by_cluster_matrix
------------------------------------------------------
A Dirichlet-multinomial is only defined when the count vector sums to its total. Reading N_e
from `cell_by_cluster_matrix` assumes sum_j y_j == clus, which holds exactly on the real atlas
but NOT on the simulations in `SIMULATION/`: those draw each junction as an independent
Binomial against the shared ATSE total, so the three junction counts of a cassette sum to that
total only in expectation (`pipelines/09_simulation/00_simulate_splicing.py:366`). Feeding a
mismatched (y, N_e) pair into the formula above yields a finite number that is not a log
probability of anything.

Deriving N_e = sum_j y_j removes the assumption entirely, costs one `index_add` we were
already paying for A_e, and matches LeafCutter's own convention of summing a cluster's
junctions to get its denominator. `denominator_consistency()` reports the gap for diagnostics.

A note on the simplex constraint
--------------------------------
If psi is simplex-constrained within each event then A_e = conc exactly, for every event and
every cell: A_e = conc * sum_{j in e} (phi_c . psi_:,j) = conc * phi_c . 1 = conc, because phi
also sums to 1. That is a genuine simplification, but it is NOT a requirement -- the DM
normalises internally, so A_e is computed by segment sum and unconstrained psi works too.
Without the constraint, conc and the within-event scale of psi are confounded (only their
product enters), so `conc` stops being separately interpretable. `check_A_e_is_conc()` reports
how close A_e sits to conc, a cheap check that the constraint is actually on.

The multinomial coefficient log(N_e! / prod_j y_j!) is omitted. It does not depend on any
parameter, so it contributes no gradient; it does mean the reported ELBO carries a constant
offset and cannot be compared numerically against a BetaBinomial run. Those are different
models, so that comparison was never meaningful anyway.
"""

import numpy as np
import torch

__all__ = ["build_atse_index", "dm_log_prob", "check_A_e_is_conc",
           "denominator_consistency", "segment_softmax"]


def build_atse_index(event_id):
    """Precompute the per-junction ATSE index.

    Parameters
    ----------
    event_id : array-like of length J
        The ATSE each junction belongs to. Any hashable labels; order is not assumed.

    Returns
    -------
    atse_idx : LongTensor [J]      contiguous ATSE id per junction, 0..n_atse-1
    rep_idx  : LongTensor [n_atse] first junction of each ATSE. Not used by `dm_log_prob`
                                   (which derives N_e from the counts); kept for diagnostics
                                   that need to read a per-event value out of a [B, J] tensor.
    sizes    : LongTensor [n_atse] number of junctions in each ATSE (the m in the docstring)
    """
    ev = np.asarray(event_id)
    _, inv = np.unique(ev, return_inverse=True)
    atse_idx = torch.as_tensor(inv, dtype=torch.long)
    n_atse = int(atse_idx.max().item()) + 1

    rep = torch.full((n_atse,), len(ev), dtype=torch.long)
    rep = rep.scatter_reduce(0, atse_idx, torch.arange(len(ev), dtype=torch.long),
                             reduce="amin", include_self=True)
    sizes = torch.zeros(n_atse, dtype=torch.long).index_add_(
        0, atse_idx, torch.ones(len(ev), dtype=torch.long))

    if int(sizes.min()) < 1:
        raise ValueError("every ATSE must contain at least one junction")
    if int(rep.max()) >= len(ev):
        raise ValueError("failed to assign a representative junction to every ATSE")
    return atse_idx, rep, sizes


def _segment_sum(x, atse_idx, n_atse):
    """Sum a [B, J] tensor within each ATSE -> [B, n_atse]."""
    out = torch.zeros(x.shape[0], n_atse, dtype=x.dtype, device=x.device)
    return out.index_add(1, atse_idx, x)


def dm_log_prob(y, mu, conc, atse_idx, n_atse, eps=1e-2, event_weight=None):
    """Per-cell Dirichlet-multinomial log-probability, summed over ATSEs.

    Parameters
    ----------
    y        : [B, J] junction counts for this minibatch
    mu       : [B, J] expected usage, phi @ psi
    conc     : scalar overdispersion; alpha = conc * mu
    atse_idx : [J] from `build_atse_index`
    n_atse   : int
    eps      : floor on alpha. See below -- this is a stability bound, not a numerical guard.

    event_weight : [n_atse] or None
        Per-event weight on the log-likelihood. None (default) is the correct likelihood.

        This is a TEMPERED pseudo-likelihood, not a probability model: weighting counts each
        event w_e times, so the posterior it targets is not the posterior of any generative
        process and its credible intervals are deliberately too narrow. It exists because the
        BetaBinomial this model replaced was doing exactly this by ACCIDENT -- scoring each of an
        ATSE's m junctions against the shared total extracts m units of information where only
        m-1 are free, an over-count of m/(m-1). Measured, that is 2.00x on the 43.6% of atlas
        ATSEs with two junctions and 1.508x junction-weighted overall, and it showed up as 95%
        intervals covering 80.7% of the truth at m=2.

        The over-count is also what gave the old model a discriminative phi: it raised the
        likelihood's pull on each cell's loading relative to the Dirichlet prior. Setting
        w_e = m/(m-1) reproduces that DELIBERATELY, so the trade can be measured and stated
        rather than inherited silently.

    Returns
    -------
    [B] log-probability per cell.

    N_e is derived as sum_{j in e} y_j, so the count vector and its total are consistent by
    construction -- see the module docstring. Events with no coverage in a cell contribute
    exactly zero without needing a mask: every y_j is 0, so the junction terms vanish and
    lgamma(A_e) - lgamma(0 + A_e) = 0.

    WHY eps IS 1e-2 AND NOT 1e-6
    ----------------------------
    d/da [lgamma(y + a) - lgamma(a)] = digamma(y + a) - digamma(a). For a junction with no reads
    the two terms cancel exactly, but for y >= 1 the -digamma(a) term survives and grows as 1/a.
    That is the atlas divergence: every atlas fit drove psi into the `clip = 1e-4` floor by step
    125, and learned conc between 0.81 and 1.9 -- an order of magnitude below the ~48 recovered on
    simulations -- so alpha = conc * mu landed at ~1e-4 and per-entry gradients reached 5,000 to
    12,000 over a [2048, 79768] tensor. ClippedAdam's clip_norm bounds the TOTAL norm, so nothing
    overflows at once; instead those entries dominate the update direction, psi is pushed harder
    into the clip and conc falls further to absorb the misfit, until a non-finite loss appears
    (observed at steps 1161/1433/1513/1663/2113/2249). Larger K fails sooner because more programs
    are driven to extreme compositions.

    A floor at 1e-6 never binds in that regime -- alpha is 190x above it -- which is why the guard
    that existed did nothing. Measured at 1e-2, on the term that actually carries the gradient
    (a clipped junction that nonetheless got a read): max |dL/da| falls from 5,249 to 96.9.

    It is a bound, not an approximation: with conc = 50 and mu >= 0.1 the floor never activates and
    the log-probability is bit-identical, including on the case verified against LeafCutter's
    dm_glm.stan. It changes only entries where the model asserts usage below eps/conc AND a read was
    observed there -- i.e. where the model is already maximally wrong, and where an unbounded
    penalty is what destroys the fit rather than correcting it.
    """
    a = (conc * mu).clamp_min(eps)                                          # [B, J]

    # Junction terms -- no grouping needed, each junction belongs to exactly one event.
    j_term = torch.lgamma(y + a) - torch.lgamma(a)                          # [B, J]

    A_e = _segment_sum(a, atse_idx, n_atse)                                 # [B, n_atse]
    N_e = _segment_sum(y, atse_idx, n_atse)                                 # [B, n_atse]
    e_term = torch.lgamma(A_e) - torch.lgamma(N_e + A_e)                    # [B, n_atse]

    if event_weight is None:
        return j_term.sum(-1) + e_term.sum(-1)
    # Weighted pseudo-likelihood. BOTH terms must carry the same weight or the event's
    # contribution stops being a scaled log-probability and becomes an arbitrary mixture.
    return (j_term * event_weight[atse_idx]).sum(-1) + (e_term * event_weight).sum(-1)


def check_A_e_is_conc(mu, conc, atse_idx, n_atse):
    """Diagnostic: how far the per-event concentration sits from `conc`.

    Returns (max_abs_rel_dev, median_abs_rel_dev). Near zero means psi is simplex-constrained
    within events, so A_e = conc and the likelihood's overdispersion is interpretable. Large
    values are not an error -- the DM still normalises -- but `conc` is then confounded with
    the within-event scale of psi.
    """
    A_e = _segment_sum(conc * mu, atse_idx, n_atse)
    rel = ((A_e - conc) / conc).abs()
    return float(rel.max()), float(rel.median())


def denominator_consistency(y, clus, atse_idx, n_atse, rep_idx):
    """Diagnostic: does sum_j y_j equal the stored ATSE total?

    Returns (max_abs_diff, frac_entries_disagreeing). Zero on the real atlas, non-zero on the
    `SIMULATION/` cassettes, which draw junctions as independent Binomials against a shared
    total. `dm_log_prob` does not depend on this holding -- it derives N_e from y -- but a
    large gap means the stored `cell_by_cluster_matrix` should not be used as a DM total.
    """
    N_from_y = _segment_sum(y, atse_idx, n_atse)
    N_stored = clus.index_select(1, rep_idx)
    d = (N_from_y - N_stored).abs()
    return float(d.max()), float((d > 1e-4).float().mean())


def segment_softmax(logit, atse_idx, n_atse, clip=1e-4):
    """Softmax within each ATSE: turns free logits into a composition summing to 1 per event.

    The max MUST be subtracted per ATSE, not globally. A global max makes every entry of every
    other event underflow to zero, that event's denominator becomes zero, and the fit diverges
    to NaN -- which is exactly what killed the first simplex run at step 1143.

    Parameters
    ----------
    logit    : [K, J] unconstrained logits
    atse_idx : [J] from `build_atse_index`
    n_atse   : int

    Returns
    -------
    [K, J] with sum_{j in e} out[k, j] == 1 for every program k and event e.
    """
    idx = atse_idx.expand(logit.shape[0], -1)
    seg_max = torch.full((logit.shape[0], n_atse), -1e30,
                         dtype=logit.dtype, device=logit.device)
    seg_max = seg_max.scatter_reduce(1, idx, logit, reduce="amax", include_self=True)
    e = torch.exp(logit - seg_max.gather(1, idx))
    denom = torch.zeros(logit.shape[0], n_atse, dtype=logit.dtype, device=logit.device)
    denom = denom.index_add(1, atse_idx, e)
    return (e / denom.gather(1, idx).clamp_min(1e-30)).clamp(clip, 1 - clip)
