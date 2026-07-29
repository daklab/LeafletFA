"""Deterministic, label-free starting points for psi.

WHY THIS EXISTS. Pyro's default (`init_to_feasible`) sets every variational mean by pushing zeros
through the constraint transform: psi = 0.5 and phi = 1/K, for every program and every cell, with zero
spread. The K programs therefore start EXACTLY identical, and the only thing that can separate them is
the ~0.02 spread of a single guide sample -- which the optimiser must amplify roughly fourfold to reach
the ~0.08 contrast a fitted model carries.

Whether that amplification lands anywhere sensible depends on how strongly the likelihood can steer it.
Under a Beta-Binomial with concentration c, the information a junction carries about the predicted rate
saturates at ~c+1 reads however deeply the cell is sequenced. At the mouse atlas's learned c ~ 1.97 the
likelihood penalises a wrong phi about twenty times less than it would with no overdispersion, so it
largely cannot steer the amplification, and each run amplifies whichever fluctuation it happened to
draw. Measured consequence in simulation at c = 2: only 30% of the achievable phi directions reproduce
between runs, and accuracy against the KNOWN phi falls to 0.699 -- worse than a crude heuristic.

WHAT THIS DOES. Start the programs apart instead of on top of each other, at a location the data
supports. Every program is centred on the observed per-junction mean and displaced by a FIXED
pseudo-random offset. Deterministic (a fixed generator seed, not the fit's seed), label-free, and
independent of K -- there is no clustering step, so nothing degenerates at large K.

An NNDSVD-based alternative was tried and rejected: its hard cell-to-component assignment is badly
unbalanced at K = 30 on the atlas, leaving 22 of 30 groups below any usable size and collapsing them
onto the global mean -- reintroducing exactly the symmetry it was meant to break.

WHAT IT DOES NOT DO. It cannot raise the information ceiling; overdispersion sets that, and nothing in
the initialisation changes it. It only lets the fit reach the ceiling instead of stopping short. And
because the offsets carry no information about the data, consistency alone is not evidence the result
is right -- always report accuracy against a known truth (in simulation) or agreement with an
independent fit (on real data) alongside it.
"""
from __future__ import annotations

import numpy as np


def observed_psi_mean(junc, clus, eps: float = 1e-8) -> np.ndarray:
    """Coverage-weighted mean observed splicing rate per junction, over covered cells only."""
    junc = np.asarray(junc, dtype=np.float32)
    clus = np.asarray(clus, dtype=np.float32)
    cov = (clus > 0).astype(np.float32)
    psi_obs = np.where(cov > 0, junc / np.maximum(clus, 1.0), 0.0)
    return (psi_obs * cov).sum(0) / np.maximum(cov.sum(0), eps)


def jitter_psi_init(junc, clus, K: int, scale: float = 0.08, seed: int = 0,
                    clip: float = 1e-3) -> np.ndarray:
    """(K, J) starting point: the data's per-junction mean, with each program displaced.

    Parameters
    ----------
    scale : float
        Standard deviation of the displacement. The default matches the ~0.08 across-program contrast
        a fitted atlas model reaches, so the programs start about as far apart as they end up rather
        than having to travel. Much smaller and the symmetry is barely broken; much larger and the
        programs start somewhere the data does not support.
    seed : int
        Generator seed for the displacement. FIXED and independent of the fit's seed -- this is the
        whole point. Passing the fit's seed here would make every run start differently and reintroduce
        the variability being removed.
    clip : float
        psi is sampled through a logit, and logit(1.0) = +15.9 sits where the sigmoid gradient is
        ~1e-7, i.e. frozen for the run. Values are kept inside [clip, 1-clip] so no entry starts dead.
    """
    mean = observed_psi_mean(junc, clus)
    off = np.random.default_rng(int(seed)).normal(0.0, float(scale), size=(int(K), mean.shape[0]))
    return np.clip(mean[None, :] + off, clip, 1.0 - clip).astype(np.float32)


def waypoint_init(junc, clus, K: int, seed: int = 42, n_pca: int = 10, n_dim: int = 10,
                  metacell_size: int = 50, clip: float = 1e-3, n_sub: int | None = None):
    """Waypoint psi/phi initialisation, from arrays rather than an AnnData.

    The scheme used before the plate rewrite, and the one the original paper describes: take an SVD of
    the centred junction-usage ratios, pick K cells by max-min sampling so they are spread through that
    space, build a metacell around each, and set program k's profile to the coverage-aware mean over
    metacell k. phi is each cell's similarity to the waypoints.

    Two properties matter for how it can be reported:

    UNSTRATIFIED. Waypoints are chosen by unsupervised max-min sampling over the SVD space. No
    cell-type label is read anywhere, so this does not compromise a claim that the programs are found
    "without prior cell-type labels" -- unlike the cell-type-stratified variant, which does.

    SEED-DEPENDENT. `max_min_sampling` draws the FIRST waypoint at random and then adds each subsequent
    one greedily. Different seeds therefore give different waypoint sets and different starting points
    -- which is what makes "three seeds, and they agree" an actual claim rather than a restatement of
    determinism. This is the opposite of `jitter_psi_init`, which is fixed by construction.

    SCALE. Centring destroys sparsity, so the SVD cannot run on the full atlas matrix (92,894 x
    79,768 is ~30 GB dense). With `n_sub` set, the waypoints and metacells are derived from a fixed
    subsample of cells and psi is the metacell mean over those cells. psi is a per-junction quantity,
    so a representative subsample gives the same profiles; phi is then returned only for the sampled
    cells (None for the rest is not useful, so phi is returned as None when subsampling -- pass psi
    alone, which is what breaks the symmetry).

    Returns (psi_init (K, J), phi_init (N, K) or None) -- the layout LeafletFAMinibatch takes,
    transposed from the (J, K) that the AnnData varm path uses.
    """
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from .waypoints import max_min_sampling, assign_nearest_cells, initialize_psi_phi

    junc = np.asarray(junc, dtype=np.float32)
    clus = np.asarray(clus, dtype=np.float32)
    n_total = junc.shape[0]
    sub = None
    if n_sub is not None and n_sub < n_total:
        # fixed subsample -- derived from `seed` so different seeds still give different waypoints,
        # which is the property the three-seed experiment depends on
        sub = np.sort(np.random.default_rng(int(seed)).choice(n_total, size=int(n_sub),
                                                              replace=False))
        junc, clus = junc[sub], clus[sub]
    mask = clus > 0
    rho = np.where(mask, junc / np.maximum(clus, 1.0), 0.0).astype(np.float32)
    # centred, so the SVD spans how cells DIFFER rather than the shared per-junction baseline, which
    # carries no information about which programs a cell uses
    Xc = np.where(mask, rho - (rho * mask).sum(0) / np.maximum(mask.sum(0), 1), 0.0)

    X_pca = TruncatedSVD(n_components=min(n_pca, min(Xc.shape) - 1),
                         random_state=seed).fit_transform(Xc)
    wp = max_min_sampling(X_pca, K, num_components=min(n_dim, X_pca.shape[1]), seed=seed)
    metacells = assign_nearest_cells(wp, X_pca, num_nearest=metacell_size)
    psi_jk, phi_nk = initialize_psi_phi(rho, metacells, mask=mask, clip=clip)

    psi = np.clip(np.asarray(psi_jk, dtype=np.float32).T, clip, 1 - clip)   # (K, J)
    assert psi.shape == (K, junc.shape[1]), f"psi {psi.shape} != {(K, junc.shape[1])}"
    if sub is not None:
        return psi, None                     # phi covers only the subsample; not useful downstream
    phi = np.asarray(phi_nk, dtype=np.float32)
    phi = np.clip(phi, 1e-6, None)
    phi = phi / phi.sum(1, keepdims=True)
    assert phi.shape == (junc.shape[0], K), f"phi {phi.shape} != {(junc.shape[0], K)}"
    return psi, phi


def phi_from_psi(junc, clus, psi, temperature: float = 0.05, chunk: int = 8192):
    """(N, K) phi init: each cell's affinity to the K program profiles.

    Needed because `waypoint_init(n_sub=...)` can only return phi for the cells it subsampled, while
    the model needs one row per cell. Deriving phi from psi instead keeps the whole atlas covered and
    costs one chunked matmul.

    A cell's affinity to program k is the negative mean squared deviation between its OBSERVED
    splicing rates and psi_k, over the junctions that cell actually covers -- so uncovered junctions
    contribute nothing rather than being scored as agreement with zero. Affinities become a simplex
    point through a softmax.

    `temperature` sets how committed the starting point is. The default is deliberately soft: the aim
    is to break the symmetry between cells, not to hand the optimiser a hard assignment it then has to
    undo. Pinning phi too confidently was measured raising seed agreement while LOWERING accuracy
    against known phi -- runs agreeing on a worse answer.
    """
    import numpy as np
    junc = np.asarray(junc, dtype=np.float32)
    clus = np.asarray(clus, dtype=np.float32)
    psi = np.asarray(psi, dtype=np.float32)                       # (K, J)
    K = psi.shape[0]
    out = np.empty((junc.shape[0], K), dtype=np.float32)
    for a in range(0, junc.shape[0], int(chunk)):
        b = min(a + int(chunk), junc.shape[0])
        cov = clus[a:b] > 0
        rho = np.where(cov, junc[a:b] / np.maximum(clus[a:b], 1.0), 0.0)
        n = np.maximum(cov.sum(1, keepdims=True), 1.0)
        # mean squared deviation over covered junctions only
        d = np.stack([(((rho - psi[k]) ** 2) * cov).sum(1, keepdims=True)[:, 0] / n[:, 0]
                      for k in range(K)], axis=1)
        z = -d / max(float(temperature), 1e-6)
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        out[a:b] = e / np.maximum(e.sum(1, keepdims=True), 1e-12)
    return out
