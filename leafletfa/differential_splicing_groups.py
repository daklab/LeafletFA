"""Group-vs-group differential splicing, following scVI's change-mode DE.

Reference: https://docs.scvi-tools.org/en/stable/user_guide/background/differential_expression.html

The scaffold is scVI's, with one substitution: the effect size is **DeltaPSI**, not a log-fold-change.
PSI is a proportion, so a ratio is the wrong scale (and undefined at 0); a difference is the natural
one, and it is what LeafCutter/rMATS report too.

For each posterior sample s and each junction j, with cell groups A and B:

    rho_scj = sum_k phi_sck * psi_skj                    # imputed (denoised) PSI for cell c
    Delta_sj = mean_{c in A} rho_scj - mean_{c in B} rho_scj
    beta_j   = (1/S) sum_s Delta_sj                      # posterior mean effect size
    P_j      = (1/S) sum_s 1( |Delta_sj| > delta )       # posterior prob. of a "real" change

and then a Bayesian FDR (posterior expected FDP) over the P_j.

ATSE-LEVEL SCORE
----------------
The junctions of one ATSE are a COMPOSITION: the data satisfy sum_{j in e} y_cj/T_cj = 1. So the
question for an event is not "did junction j move?" but "how much splicing mass moved within e?" --
i.e. the total variation distance between the group's junction-usage distributions:

    TV_se = 0.5 * sum_{j in e} | mean_A rho_sj - mean_B rho_sj |
          = 0.5 * sum_{j in e} | Delta_sj |

TV is the right statistic here because (a) it respects the constraint that a gain on one junction
must be paid for by a loss on another, (b) for a two-junction ATSE it reduces exactly to |DeltaPSI|,
so it agrees with the junction-level test, and (c) delta_atse is interpretable ("10% of the mass
moved").

**TV is computed PER POSTERIOR SAMPLE and only then thresholded** -- never on the posterior mean.
Taking TV of the mean would discard exactly the uncertainty the posterior-sampling scaffold exists
to propagate.

CAVEAT worth stating in the Methods: LeafletFA does not *enforce* the simplex constraint within an
ATSE. Each psi_kj ~ Beta(a_j, b_j) independently and the likelihood scores each junction as its own
Binomial with T_cj trials, so sum_{j in e} rho_cj ~= 1 only insofar as the model learns it. Use
`atse_simplex_check` to verify it holds on your fit before trusting TV.
"""

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Bayesian FDR
# ---------------------------------------------------------------------------

def bayesian_fdr(probs, alpha=0.05):
    """Posterior expected FDP, thresholded at the MOST PERMISSIVE tau (== max detections).

    Rank by P descending; the expected false discovery proportion among the top n is

        FDP(n) = (1/n) * sum_{i <= n} (1 - P_(i))

    and we take the LARGEST n with FDP(n) <= alpha. This is scVI's rule.

    NB the earlier implementation used `len(np.where(fdp <= alpha)[0])`, which counts *how many*
    ranks satisfy the bound rather than finding the last one. Those agree only if the satisfying set
    is a prefix. FDP(n) is a running mean of (1 - P), which is non-decreasing in n only when P is
    sorted -- ties and plateaus can make it dip back under alpha, and then `len()` silently
    undercounts. Use `argmax(...) + 1`.

    Returns
    -------
    significant : bool array, in the ORIGINAL order
    fdp         : float array, the FDP at each item's own rank, in the ORIGINAL order
    n_sig       : int
    """
    probs = np.asarray(probs, dtype=np.float64)
    n = len(probs)
    if n == 0:
        return np.zeros(0, bool), np.zeros(0), 0

    order = np.argsort(-probs)                       # descending P
    p_sorted = probs[order]
    fdp_sorted = np.cumsum(1.0 - p_sorted) / np.arange(1, n + 1)

    ok = np.flatnonzero(fdp_sorted <= alpha)
    n_sig = int(ok.max() + 1) if ok.size else 0      # <-- most permissive tau

    significant = np.zeros(n, dtype=bool)
    significant[order[:n_sig]] = True

    fdp = np.empty(n)
    fdp[order] = fdp_sorted
    return significant, fdp, n_sig


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def group_delta_psi(psi_samples, phi_samples, cells_1, cells_2):
    """Per-posterior-sample DeltaPSI for EVERY junction at once. Returns [S, J].

    rho is LINEAR in phi, so

        mean_{c in A} rho_scj = ( mean_{c in A} phi_sck ) @ psi_skj

    i.e. we can average phi over the group FIRST and never materialise the [S, C, J] tensor. At
    S=100, C=100k, J=80k that tensor would be 3 PB; this is [S, K] @ [S, K, J].
    """
    psi_samples = torch.as_tensor(psi_samples)
    phi_samples = torch.as_tensor(phi_samples)

    phibar_1 = phi_samples[:, cells_1, :].mean(dim=1)          # [S, K]
    phibar_2 = phi_samples[:, cells_2, :].mean(dim=1)          # [S, K]
    dphi = phibar_1 - phibar_2                                  # [S, K]
    return torch.einsum("sk,skj->sj", dphi, psi_samples)        # [S, J]


def group_phibar_from_q(phi_loc, phi_scale, cells, n_samples, seed=0):
    """Posterior draws of a group's MEAN phi, [S, K], from AutoNormal's q(phi).

    phi is Dirichlet, so AutoNormal parameterises it in unconstrained space through a
    stick-breaking transform and the stored trailing dimension is K-1, not K.

    Only the group's own cells are drawn. The full [S, C, K] tensor is 2.1 GB at S=200, C=132,706,
    K=20, and every cell outside the contrast is irrelevant to it -- so this keeps
    `group_delta_psi`'s maths unchanged while making phi sampling affordable across 73 contrasts.

    WHY THIS MATTERS. A26 fixed phi at its posterior mean because the 2026-07-26 atlas fit carried
    no phi posterior -- it was never written, not lost. With phi fixed, ALL the width in the effect
    size came from psi, so the test could look confident while the per-cell encoding underneath it
    was unreliable. That is precisely the failure mode the atlas turned out to have, and it is
    invisible unless phi is sampled.
    """
    loc = torch.as_tensor(np.asarray(phi_loc, np.float32))
    scale = torch.as_tensor(np.asarray(phi_scale, np.float32))
    if loc.numel() == 0:
        raise ValueError("fit carries no q(phi) -- retrain with a version of A44 that saves it")
    cells = np.asarray(cells, dtype=np.int64)
    lo, sc = loc[cells], scale[cells]
    sb = torch.distributions.transforms.StickBreakingTransform()
    g = torch.Generator().manual_seed(int(seed))
    out = torch.empty(n_samples, lo.shape[0] if lo.dim() == 1 else sb(lo[:1]).shape[-1])
    for s_i in range(n_samples):
        eps = torch.randn(lo.shape, generator=g)
        out[s_i] = sb(lo + sc * eps).mean(0)          # [K] -- the group mean for this draw
    return out


def group_delta_psi_from_q(psi_samples, phi_loc, phi_scale, cells_1, cells_2, seed=0):
    """As `group_delta_psi`, but drawing phi from q(phi) rather than taking it fixed. [S, J]."""
    psi_samples = torch.as_tensor(psi_samples)
    S = psi_samples.shape[0]
    p1 = group_phibar_from_q(phi_loc, phi_scale, cells_1, S, seed=seed)
    p2 = group_phibar_from_q(phi_loc, phi_scale, cells_2, S, seed=seed + 1)
    return torch.einsum("sk,skj->sj", p1 - p2, psi_samples)


def _resolve_groups(adata, group_1, group_2, groupby):
    labels = adata.obs[groupby].values
    c1 = np.flatnonzero(labels == group_1)
    if len(c1) == 0:
        raise ValueError(f"no cells with {groupby} == {group_1!r}")
    c2 = np.flatnonzero(labels == group_2) if group_2 is not None else np.flatnonzero(labels != group_1)
    if len(c2) == 0:
        raise ValueError(f"no cells in the comparison group")
    return c1, c2


def differential_splicing_junctions(adata, psi_samples, phi_samples, group_1, group_2=None,
                                    groupby="cell_type", delta=0.1, alpha=0.05):
    """Junction-level DS between two groups of cells. Vectorised over all junctions.

    Parameters
    ----------
    psi_samples : [S, K, J] posterior samples of psi
    phi_samples : [S, C, K] posterior samples of phi
    delta       : effect-size threshold on |DeltaPSI| (scVI's "change mode" delta)
    alpha       : target posterior expected FDP

    Returns a DataFrame indexed like adata.var, with beta (posterior mean DeltaPSI), prob_de,
    lfdr (= 1 - prob_de), fdp, and `significant`.
    """
    c1, c2 = _resolve_groups(adata, group_1, group_2, groupby)
    d = group_delta_psi(psi_samples, phi_samples, c1, c2)       # [S, J]

    beta = d.mean(dim=0).cpu().numpy()
    prob_de = (d.abs() > delta).float().mean(dim=0).cpu().numpy()

    # delta is ALREADY inside prob_de, so do NOT also filter on |beta| > delta afterwards -- doing so
    # changes the composition of the reported set after its FDP was computed, and the reported FDR
    # then does not describe the set you report.
    significant, fdp, n_sig = bayesian_fdr(prob_de, alpha=alpha)

    print(f"[junction DS] {group_1} vs {group_2 or 'rest'}: "
          f"{n_sig:,} / {len(beta):,} junctions at FDP<={alpha} (delta={delta})")

    return pd.DataFrame({
        "beta": beta,
        "abs_beta": np.abs(beta),
        "prob_de": prob_de,
        "lfdr": 1.0 - prob_de,
        "fdp": fdp,
        "significant": significant,
        "n_cells_1": len(c1),
        "n_cells_2": len(c2),
    }, index=adata.var_names)


def differential_splicing_atses(adata, psi_samples, phi_samples, group_1, group_2=None,
                                groupby="cell_type", event_key="event_id",
                                delta=0.1, alpha=0.05):
    """ATSE-level DS: total variation of the junction-usage composition, per posterior sample.

    TV_se = 0.5 * sum_{j in e} |Delta_sj|,  computed for EACH posterior sample s, then thresholded.
    """
    c1, c2 = _resolve_groups(adata, group_1, group_2, groupby)
    d = group_delta_psi(psi_samples, phi_samples, c1, c2)       # [S, J]

    events = adata.var[event_key].astype(str).values
    uniq, inv = np.unique(events, return_inverse=True)
    inv_t = torch.as_tensor(inv, dtype=torch.long, device=d.device)

    # TV per (sample, ATSE): 0.5 * sum of |Delta| over the junctions of the event
    S = d.shape[0]
    tv = torch.zeros((S, len(uniq)), dtype=d.dtype, device=d.device)
    tv.index_add_(1, inv_t, d.abs())
    tv = 0.5 * tv                                                # [S, n_atse]

    tv_mean = tv.mean(dim=0).cpu().numpy()
    prob_de = (tv > delta).float().mean(dim=0).cpu().numpy()
    significant, fdp, n_sig = bayesian_fdr(prob_de, alpha=alpha)

    print(f"[ATSE DS] {group_1} vs {group_2 or 'rest'}: "
          f"{n_sig:,} / {len(uniq):,} ATSEs at FDP<={alpha} (delta_TV={delta})")

    return pd.DataFrame({
        "event_id": uniq,
        "tv": tv_mean,             # posterior mean total variation (the effect size)
        "prob_de": prob_de,
        "lfdr": 1.0 - prob_de,
        "fdp": fdp,
        "significant": significant,
    }).set_index("event_id")


def atse_simplex_check(adata, psi_samples, phi_samples, event_key="event_id", n_cells=2000):
    """Do the imputed PSIs actually sum to ~1 within each ATSE?

    LeafletFA does NOT enforce this: psi_kj ~ Beta(a_j, b_j) independently, and the likelihood scores
    each junction as its own Binomial. So the composition holds only insofar as the model learns it.
    TV assumes it. Check before trusting ATSE-level scores.
    """
    psi = torch.as_tensor(psi_samples).mean(dim=0)               # [K, J]
    phi = torch.as_tensor(phi_samples).mean(dim=0)               # [C, K]
    idx = np.random.default_rng(0).choice(phi.shape[0], min(n_cells, phi.shape[0]), replace=False)
    rho = (phi[idx] @ psi).cpu().numpy()                         # [n, J]

    events = adata.var[event_key].astype(str).values
    uniq, inv = np.unique(events, return_inverse=True)
    sums = np.zeros((rho.shape[0], len(uniq)))
    np.add.at(sums.T, inv, rho.T)

    print(f"[simplex check] sum of imputed PSI within an ATSE (should be ~1.0):")
    print(f"   mean {sums.mean():.3f}   median {np.median(sums):.3f}   "
          f"5-95% [{np.percentile(sums, 5):.3f}, {np.percentile(sums, 95):.3f}]")
    return sums
