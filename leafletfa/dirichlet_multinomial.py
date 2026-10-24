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

__all__ = ["build_atse_index", "dm_log_prob", "dm_log_prob_sparse", "SparseObs", "check_A_e_is_conc",
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


# ── sparse evaluation over observed events ─────────────────────────────────────────────────
# The dense dm_log_prob above touches every (cell, junction) entry, but an ATSE with no reads in a
# cell contributes exactly zero (every y_j = 0, and lgamma(A_e) - lgamma(0 + A_e) = 0). On TMS a
# cell observes ~6.5% of entries, so ~94% of the dense work -- and of the 35 GB of dense matrices
# held for a 55k-cell fit -- computes zeros. The functions below evaluate the SAME log-probability
# on the observed events only: every junction of every event with at least one read in the cell.
# Same terms, same eps floor, same event weights; only the summation order differs.

class SparseObs:
    """Per-cell observed-event entries of a junction count matrix, stored CSR-like on `device`.

    For cell c the entries are all junctions of every event with sum_j y[c, j] > 0, so N_e > 0
    for every segment and no event that the dense likelihood scores non-trivially is dropped.
    Observedness is taken from y itself, consistent with dm_log_prob deriving N_e from y.

    y        : scipy.sparse (N, J) junction counts (any non-negative values)
    atse_idx : LongTensor [J] from build_atse_index
    """

    def __init__(self, y, atse_idx, device="cpu"):
        import scipy.sparse as sp
        y = sp.csr_matrix(y, dtype=np.float32)
        y.eliminate_zeros()
        y.sort_indices()
        N, J = y.shape
        ev = np.asarray(atse_idx.cpu().numpy() if torch.is_tensor(atse_idx) else atse_idx,
                        dtype=np.int64)
        n_atse = int(ev.max()) + 1
        # junctions grouped by event
        order = np.argsort(ev, kind="stable")
        bounds = np.searchsorted(ev[order], np.arange(n_atse + 1))
        size = np.diff(bounds)
        S = sp.csr_matrix((np.ones(J, np.float32), (np.arange(J), ev)), shape=(J, n_atse))
        # Built in blocks of cells so the int64 temporaries stay bounded (~0.7e9 entries on Allen
        # would otherwise peak at ~50-60 GB); only the int32/float32 results are kept.
        cols_l, vals_l, segl_l, ent_n, seg_ptr_l = [], [], [], [], [np.zeros(1, np.int64)]
        step = max(1, int(2e7 // max(1, (y.nnz / max(N, 1)) * 3)))
        for r0 in range(0, N, step):
            yb = y[r0:r0 + step]
            nb = yb.shape[0]
            E = (yb @ S).tocsr()                                    # observed (cell, event) pairs
            E.eliminate_zeros()
            E.sort_indices()
            seg_ev = E.indices.astype(np.int64)
            seg_cell = np.repeat(np.arange(nb), np.diff(E.indptr))
            seg_len = size[seg_ev]
            n_b = int(seg_len.sum())
            seg_of_ent = np.repeat(np.arange(len(seg_ev)), seg_len)
            first = np.cumsum(seg_len) - seg_len
            cols = order[bounds[seg_ev][seg_of_ent] + (np.arange(n_b) - first[seg_of_ent])]
            rows = seg_cell[seg_of_ent]
            # y at those entries: look up in the block's CSR by (row, col) key
            key_y = np.repeat(np.arange(nb, dtype=np.int64), np.diff(yb.indptr)) * J + yb.indices
            key_e = rows.astype(np.int64) * J + cols
            pos = np.minimum(np.searchsorted(key_y, key_e), max(len(key_y) - 1, 0))
            hit = (key_y[pos] == key_e) if len(key_y) else np.zeros(n_b, bool)
            vals = np.where(hit, yb.data[pos] if len(key_y) else 0.0, 0.0).astype(np.float32)
            del key_y, key_e, pos, hit
            cols_l.append(cols.astype(np.int32)); vals_l.append(vals)
            segl_l.append((seg_of_ent - E.indptr[rows]).astype(np.int32))
            ent_n.append(np.bincount(rows, minlength=nb))
            seg_ptr_l.append(E.indptr[1:].astype(np.int64) + seg_ptr_l[-1][-1])
        cols, vals, seg_local = (np.concatenate(x) if x else np.zeros(0, dt) for x, dt in
                                 ((cols_l, np.int32), (vals_l, np.float32), (segl_l, np.int32)))
        n_ent = len(cols)
        assert np.isclose(vals.sum(dtype=np.float64), y.data.sum(dtype=np.float64), rtol=1e-9), \
            "observed entries lost reads"
        ent_ptr = np.concatenate([[0], np.cumsum(np.concatenate(ent_n) if ent_n else [])]).astype(np.int64)
        seg_ptr = np.concatenate(seg_ptr_l)

        t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
        self.N, self.J, self.n_atse = N, J, n_atse
        self.ent_ptr = t(ent_ptr, torch.long)
        self.cols = t(cols, torch.int32)
        self.y = t(vals, torch.float32)
        self.seg_local = t(seg_local, torch.int32)
        self.seg_ptr = t(seg_ptr, torch.long)
        self.n_entries = n_ent

    def nbytes(self):
        return sum(int(x.numel() * x.element_size()) for x in
                   (self.ent_ptr, self.cols, self.y, self.seg_local, self.seg_ptr))

    def batch(self, idx, device=None):
        """Gather the entries of cells `idx` (LongTensor [B]). Returns
        (rows [nnz] local row 0..B-1, cols [nnz], y [nnz], seg [nnz], seg_row [n_seg], n_seg)."""
        dev = device or self.y.device
        idx = idx.to(self.ent_ptr.device)
        B = idx.numel()
        ar = torch.arange(B, device=idx.device)
        s, e = self.ent_ptr[idx], self.ent_ptr[idx + 1]
        L = e - s
        rows = torch.repeat_interleave(ar, L)
        excl = torch.cumsum(L, 0) - L
        pos = torch.repeat_interleave(s - excl, L) + torch.arange(int(L.sum()), device=idx.device)
        nseg = self.seg_ptr[idx + 1] - self.seg_ptr[idx]
        seg_off = torch.cumsum(nseg, 0) - nseg
        seg = seg_off[rows] + self.seg_local[pos].long()
        seg_row = torch.repeat_interleave(ar, nseg)
        out = (rows, self.cols[pos].long(), self.y[pos], seg, seg_row, int(nseg.sum()))
        return tuple(x.to(dev, non_blocking=True) if torch.is_tensor(x) else x for x in out)


class _SampledMatmul(torch.autograd.Function):
    """out[n] = phi[rows[n]] . psiT[cols[n]]  -- (phi @ psi) evaluated only at the given entries.

    The product itself is cheap on BLAS (K is small); what made the dense path slow was the
    likelihood over every entry. So each block of rows is multiplied densely and only the observed
    entries are gathered; the backward scatters the entry gradients into the same dense block and
    goes back through two matmuls. `rows` must be sorted ascending (SparseObs.batch guarantees it).
    Peak extra memory is one (ROWS x J) block."""
    ROWS = 1024

    @staticmethod
    def _blocks(rows, B):
        bounds = torch.searchsorted(rows, torch.arange(0, B + _SampledMatmul.ROWS, _SampledMatmul.ROWS,
                                                       device=rows.device).clamp_max(B))
        for i in range(len(bounds) - 1):
            r0 = i * _SampledMatmul.ROWS
            r1 = min(r0 + _SampledMatmul.ROWS, B)
            if r0 >= B:
                break
            yield r0, r1, int(bounds[i]), int(bounds[i + 1])

    @staticmethod
    def forward(ctx, phi, psiT, rows, cols):
        B, J = phi.shape[0], psiT.shape[0]
        out = torch.empty(rows.numel(), dtype=phi.dtype, device=phi.device)
        for r0, r1, a, b in _SampledMatmul._blocks(rows, B):
            if b > a:
                M = phi[r0:r1] @ psiT.t()                                   # [rb, J]
                out[a:b] = M.reshape(-1).index_select(0, (rows[a:b] - r0) * J + cols[a:b])
        ctx.save_for_backward(phi, psiT, rows, cols)
        return out

    @staticmethod
    def backward(ctx, g):
        phi, psiT, rows, cols = ctx.saved_tensors
        B, J = phi.shape[0], psiT.shape[0]
        gphi = torch.zeros_like(phi) if ctx.needs_input_grad[0] else None
        gpsi = torch.zeros_like(psiT) if ctx.needs_input_grad[1] else None
        for r0, r1, a, b in _SampledMatmul._blocks(rows, B):
            if b <= a:
                continue
            G = torch.zeros((r1 - r0) * J, dtype=g.dtype, device=g.device)
            G.index_add_(0, (rows[a:b] - r0) * J + cols[a:b], g[a:b])
            G = G.view(r1 - r0, J)
            if gphi is not None:
                gphi[r0:r1] = G @ psiT
            if gpsi is not None:
                gpsi.addmm_(G.t(), phi[r0:r1])
        return gphi, gpsi, None, None


def dm_log_prob_sparse(batch, phi, psi, conc, atse_idx, clip=1e-4, eps=1e-2, event_weight=None):
    """dm_log_prob on observed events only. Equal to
    dm_log_prob(y_dense, (phi @ psi).clamp(clip, 1 - clip), conc, ...) up to summation order.

    batch : output of SparseObs.batch(idx)
    phi   : [B, K];  psi : [K, J];  conc : [J] per-junction concentration or a scalar
    Returns [B] log-probability per cell.
    """
    rows, cols, y, seg, seg_row, n_seg = batch
    B = phi.shape[0]
    mu = _SampledMatmul.apply(phi, psi.t().contiguous(), rows, cols).clamp(clip, 1 - clip)
    c = conc[cols] if (torch.is_tensor(conc) and conc.ndim == 1) else conc
    a = (c * mu).clamp_min(eps)
    j_term = torch.lgamma(y + a) - torch.lgamma(a)
    A_e = torch.zeros(n_seg, dtype=a.dtype, device=a.device).index_add(0, seg, a)
    N_e = torch.zeros(n_seg, dtype=a.dtype, device=a.device).index_add(0, seg, y)
    e_term = torch.lgamma(A_e) - torch.lgamma(N_e + A_e)
    if event_weight is not None:
        w = event_weight[atse_idx[cols]]
        j_term = j_term * w
        w_seg = torch.zeros(n_seg, dtype=w.dtype, device=w.device).index_copy(0, seg, w)
        e_term = e_term * w_seg
    out = torch.zeros(B, dtype=a.dtype, device=a.device)
    return out.index_add(0, rows, j_term).index_add(0, seg_row, e_term)
