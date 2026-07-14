import warnings

import numpy as np 
import pandas as pd
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import anndata as ad
import matplotlib.pyplot as plt

try:
    import scanpy as sc
    _SCANPY_AVAILABLE = True
except ImportError:
    _SCANPY_AVAILABLE = False

def sparse_sum(x, dim):
    """
    Compute the sum of a sparse matrix along a specified dimension and return a squeezed array.

    Parameters:
    x (spmatrix): A sparse matrix whose elements are to be summed.
    dim (int): The dimension along which the sum is computed. For example, `dim=0` sums along the rows, 
               while `dim=1` sums along the columns.

    Returns:
    ndarray: The resulting dense array with the sums, with single-dimensional entries removed from its shape.
    """
    return np.squeeze(np.asarray(x.sum(dim)))

def max_min_sampling(diffmap_components, n_waypoints, num_components=10, seed=42):

    # Set a random seed for reproducibility
    np.random.seed(seed)

    # Use only the first `num_components` diffusion components
    X_diff = diffmap_components[:, :num_components]
    
    # Step 1: Randomly initialize the first waypoint
    first_waypoint = np.random.choice(X_diff.shape[0], 1)[0]
    waypoints = [int(first_waypoint)]
    
    # Step 2: Initialize distance array to track the minimum distance from each cell to the nearest waypoint
    distances = np.linalg.norm(X_diff - X_diff[first_waypoint, :], axis=1)
    
    # Step 3: Iteratively add waypoints
    for _ in range(1, n_waypoints):
        # Find the cell with the maximum distance to the nearest waypoint
        next_waypoint = np.argmax(distances)
        waypoints.append(next_waypoint)
        
        # Update the minimum distances for all cells based on the new waypoint
        new_distances = np.linalg.norm(X_diff - X_diff[next_waypoint, :], axis=1)
        distances = np.minimum(distances, new_distances)
    
    return np.array(waypoints)

def initialize_psi_phi(rho_hat, metacell_dict, mask=None, epsilon=1e-3, clip=1e-4):
    """Initialize psi/phi from waypoint metacells: the COVERAGE-AWARE MEAN over each metacell.

    psi[j, k] = mean of rho_hat[c, j] over the cells c in metacell k where junction j's ATSE is
    actually COVERED. Coverage comes from `mask`, not from `rho_hat != 0`.

    Why this matters (this replaces `initialize_psi_phi_with_random_non_zero`, which was wrong in
    three compounding ways -- measured on the real 132,706 x 79,768 mouse training matrix):

    1. It took ONE uniformly random non-zero value per junction (`np.random.choice`), not a mean.
       A psi column was therefore a random mosaic of single cells, not a denoised profile -- the
       exact opposite of the Methods' stated rationale ("to reduce noise").
    2. It treated every observed ZERO as missing (`arr > 0`). But **58.6% of stored junc_ratio
       values are exactly 0.0** -- a junction genuinely unused in a covered ATSE. Dropping them
       biased the init upward: the sampled pool has mean 0.735 where the coverage-aware mean is
       0.305.
    3. It never clipped. **51.7% of the values it could draw were exactly 1.0**, and Pyro maps a
       Beta site through a logit, so psi_init = 1.0 becomes a guide loc of +15.94 where
       sigmoid'(loc) ~= 1.2e-7. About HALF the psi matrix started frozen in the flat tail of the
       sigmoid and never moved. Waypoints exist to break the symmetry (see the saddle-point
       discussion in METHODS.md §11.1); as written, they could not.

    Parameters
    ----------
    rho_hat : (n_cells, n_junctions) array or scipy sparse
        Observed junction usage ratios y/T. Kept sparse -- never densified.
    metacell_dict : dict
        waypoint -> list of cell indices (from `assign_nearest_cells`).
    mask : (n_cells, n_junctions) array or scipy sparse, optional
        1 where the junction's ATSE has reads (i.e. rho_hat is OBSERVED, zero or not), 0 where it is
        missing. Build it as `cell_by_cluster_matrix > 0`, or use the `psi_mask` layer if present.
        If omitted, falls back to treating non-zeros as observed and WARNS -- that is the old,
        biased behaviour.
    epsilon : float
        Value used for a junction with no coverage anywhere in the metacell.
    clip : float
        psi is clipped to [clip, 1 - clip] so the logit does not saturate.

    Returns
    -------
    psi : (n_junctions, n_metacells)
    phi : (n_cells, n_metacells), rows sum to 1
    """
    rho = csr_matrix(rho_hat) if not isinstance(rho_hat, csr_matrix) else rho_hat

    if mask is None:
        warnings.warn(
            "initialize_psi_phi: no coverage `mask` given, so observed zeros cannot be distinguished "
            "from missing data. Falling back to 'non-zero == observed', which biases psi_init upward "
            "(measured mean 0.735 vs 0.305 on the mouse atlas). Pass mask=(cluster_counts > 0).",
            RuntimeWarning,
        )
        cover = (rho != 0)
    else:
        cover = csr_matrix(mask)
        cover = (cover != 0)
    cover = csr_matrix(cover, dtype=np.float32)

    n_junctions = rho.shape[1]
    n_metacells = len(metacell_dict)
    psi = np.full((n_junctions, n_metacells), epsilon, dtype=np.float64)

    for i, cell_indices in enumerate(tqdm(list(metacell_dict.values()), desc="waypoint psi")):
        idx = np.asarray(cell_indices, dtype=int)
        # sum of observed rho, and count of covered cells, per junction -- both over ~50 rows only
        tot = np.asarray(rho[idx].sum(axis=0)).ravel()
        n_cov = np.asarray(cover[idx].sum(axis=0)).ravel()
        covered = n_cov > 0
        psi[covered, i] = tot[covered] / n_cov[covered]     # <-- MEAN over covered cells

    # Keep psi strictly inside (0,1): logit(1.0) saturates and freezes the entry for the whole run.
    psi = np.clip(psi, clip, 1.0 - clip)

    # phi by least squares, rho_hat ~= phi @ psi.T, solved via the normal equations so we never
    # materialise rho_hat.T densely (that was 132,706 x 79,768 float64 = 85 GB, and is why the SLURM
    # job needed 300 GB and why waypoints did not scale).
    #   phi = (rho @ psi) @ inv(psi.T @ psi)
    G = psi.T @ psi                                   # (K, K)
    B = np.asarray(rho @ psi)                         # (C, K) -- sparse x dense, cheap
    phi = B @ np.linalg.pinv(G)                       # (C, K)

    phi = np.clip(phi, epsilon, 1.0 - epsilon)
    phi /= phi.sum(axis=1, keepdims=True)
    return psi, phi


def initialize_psi_phi_with_random_non_zero(rho_hat, metacell_dict, epsilon=0.01):
    """DEPRECATED. Kept only so old scripts do not break; forwards to `initialize_psi_phi`.

    The original implementation is described (and its three bugs quantified) in
    `initialize_psi_phi`. Do not use it for new runs -- and pass a coverage `mask`.
    """
    warnings.warn(
        "initialize_psi_phi_with_random_non_zero is deprecated: it seeded psi with ONE RANDOM "
        "non-zero value per junction, treated observed zeros as missing, and left ~52% of entries "
        "at exactly 1.0 (a saturated logit, gradient ~1e-7, frozen for the whole run). "
        "Use initialize_psi_phi(rho_hat, metacell_dict, mask=cluster_counts > 0).",
        DeprecationWarning, stacklevel=2,
    )
    return initialize_psi_phi(rho_hat, metacell_dict, mask=None, epsilon=epsilon)

def assign_nearest_cells(waypoints, diffmap_components, num_nearest=50):
    """Assign the `num_nearest` closest cells to each waypoint, without reuse.

    Default is 50 to match the Methods ("the average ... across the 50 nearest neighbors"). It was
    5, and the pipeline passed 30 -- three different values, none of them the documented one.

    NB: cells are consumed greedily, so later waypoints get progressively more distant "nearest"
    neighbours. That is not in the Methods but it is deliberate (metacells must be disjoint).
    """
    X_diff = diffmap_components
    assigned_cells = set()  # To keep track of already assigned cells
    metacell_dict = {}  # Dictionary to store metacell groupings

    for i, waypoint in enumerate(waypoints):
        # Compute distances from the waypoint to all cells
        distances = np.linalg.norm(X_diff - X_diff[waypoint, :], axis=1)
        
        # Sort cells by distance, and select the nearest ones excluding already assigned cells
        sorted_indices = np.argsort(distances)
        nearest_cells = [idx for idx in sorted_indices if idx not in assigned_cells][:num_nearest]
        
        # Assign the cells to the current metacell
        metacell_dict[f"metacell_{i}"] = nearest_cells
        
        # Add these cells to the set of assigned cells
        assigned_cells.update(nearest_cells)
    
    return metacell_dict

def generate_initializations(rho_hat, waypoints_dict, metacell_dicts, mask=None, epsilon=1e-3):
    """
    Generate multiple psi and phi initializations based on different waypoint sets and metacell dictionaries.

    Parameters:
    rho_hat (ndarray): Observed probabilities (n_cells, n_junctions).
    waypoints_dict (dict): Dictionary with keys as waypoint sizes and values as waypoint sets.
    metacell_dicts (dict): Dictionary with keys as waypoint sizes and values as metacell dictionaries.
    epsilon (float): Small value to use as fallback if all junction values are zero.

    Returns:
    psi_list (list): List of psi initializations.
    phi_list (list): List of phi initializations.
    """

    psi_list = []  # To store psi initializations
    phi_list = []  # To store phi initializations

    # Iterate over each set of waypoints and corresponding metacell dictionary
    for n_waypoints in waypoints_dict.keys():
        
        print(f"Finding {n_waypoints} waypoints from the diffusion components!")

        waypoints = waypoints_dict[n_waypoints]
        metacell_dict = metacell_dicts[n_waypoints]

        # Initialize psi and phi for the current metacell dictionary.
        # `mask` marks where rho_hat is OBSERVED (ATSE covered) rather than missing -- without it,
        # observed zeros are silently dropped and psi_init is biased upward. Pass
        # mask = (cell_by_cluster_matrix > 0).
        psi, phi = initialize_psi_phi(rho_hat, metacell_dict, mask=mask, epsilon=epsilon)

        # Store the initialization
        psi_list.append(psi)
        phi_list.append(phi)

    return psi_list, phi_list

def calculate_centered_psi(junction_counts, cluster_counts, rho=0.1):
    """
    Calculates centered PSI values and other related matrices based on the input junction counts and cluster counts.
    
    Weighting is used to:
    - Account for variability in junction counts using the beta-binomial variance model.
    - Adjust the PSI values based on the reliability of the cluster (or junction) counts.

    Parameters:
    - junction_counts: A sparse matrix (COO format) with junction counts.
    - cluster_counts: A sparse matrix (COO format) with cluster counts (ATSE counts).
    - rho: A float value (default: 0.1) representing the overdispersion parameter for the beta-binomial variance model.
    
    Returns:
    - Y_sparse: A sparse matrix with centered PSI values.
    - psi: A sparse matrix with calculated PSI values.
    - w_psi: A sparse matrix with weighted PSI values.
    - junc_means: A dense array with the mean junction usage ratios.
    """
    
    # Step 1: Calculate PSI (junction usage ratios) only for non-zero cluster counts
    non_zero_mask = cluster_counts.data != 0
    junc_ratio_data = junction_counts.data[non_zero_mask] / cluster_counts.data[non_zero_mask]

    psi = coo_matrix((junc_ratio_data, (junction_counts.row[non_zero_mask], junction_counts.col[non_zero_mask])),
                     shape=junction_counts.shape)  # Create a sparse matrix with calculated PSI values

    # Step 2: Calculate observation weights (w) using the beta-binomial variance model
    # Downweights low-depth junctions where PSI estimates are noisy
    # 𝑤𝑖𝑗=precision of the PSI𝑖𝑗, which comes from the inverse of the beta-binomial variance.
    # so doing weighted mean instead of regular mean to account for differences in atse counts and how that could affect PSI estimates...
    # we are trying to prevent High activity in latent components driven by expression differences, not splicing patterns

    w = junction_counts.copy()  # Observation weights are based on junction counts
    w.data = cluster_counts.data / (1. + (cluster_counts.data - 1) * rho)  # Beta-binomial variance model

    # Step 3: Calculate weighted PSI (w_psi)
    w_psi = w.copy()  # Copy weight matrix
    w_psi.data *= psi.data  # Multiply weights by PSI data

    # Step 4: Calculate mean junction usage ratios (junc_means)
    junc_means = sparse_sum(w_psi, 0) / sparse_sum(w, 0)  # Weighted mean junction usage ratio per junction

    # Step 5: Center Y by subtracting the mean junction usage ratio from each PSI value
    Y_data = psi.data - junc_means[psi.col]  # Center PSI values

    # Step 6: Create a sparse matrix for Y_data using the row, col indices of the original psi matrix
    Y_sparse = coo_matrix((Y_data, (psi.row, psi.col)), shape=psi.shape)
    
    return Y_sparse

def plot_PCA_with_waypoints(adata, waypoints_dict, color_by='tissue', n_waypoints=20, 
                            save_plot=False,
                            waypoint_color='red', first_waypoint_color='blue', size=10):
    """
    Plots PCA with cells colored by a specified annotation (e.g., 'tissue', 'cell_type') 
    and overlays waypoints on top.
    """

    # Check if waypoints exist
    waypoints = waypoints_dict.get(n_waypoints)
    if waypoints is None:
        raise ValueError(f"No waypoints found for {n_waypoints}. Please check your waypoints_dict.")
    
    # Extract PCA coordinates
    pca_coords = adata.obsm['X_pca']
    waypoint_coords_pca = pca_coords[waypoints]

    # Create scatter plot
    if not _SCANPY_AVAILABLE:
        raise ImportError("scanpy is required for PCA plotting. Install it with: pip install scanpy")
    sc.pl.pca(adata, color=color_by, size=size, show=False)  # Standard PCA plot

    # Overlay waypoints
    plt.scatter(waypoint_coords_pca[1:, 0], waypoint_coords_pca[1:, 1], s=40, 
                color=waypoint_color, edgecolors='black', label=f'{n_waypoints} Waypoints')

    plt.scatter(waypoint_coords_pca[0, 0], waypoint_coords_pca[0, 1], s=80, 
                color=first_waypoint_color, edgecolors='black', label='First Waypoint')

    # Add legend
    plt.legend()
    plt.show()

    # Save and show plot
    if save_plot:
        filename = f"PCA_with_waypoints_{n_waypoints}_waypoints.pdf"
        plt.savefig(filename, format='pdf')
    


def add_waypoint_initializations(adata, K, n_pca=10, n_dim=10, metacell_size=50, seed=42,
                                 layer=None, mask_layer=None):
    """Compute waypoint psi/phi initialisations and WRITE THEM INTO `adata`.

    THIS IS THE GLUE THAT WAS MISSING FROM THE PACKAGE.

    `LeafletFA(waypoints_use=True)` only *reads* `adata.varm[f"psi_init_{K}_waypoints"]` and
    `adata.obsm[f"phi_init_{K}_waypoints"]` -- and raises ValueError if they are absent. But nothing
    in the package ever WROTE them: the code that builds them lived only inside the analysis repo's
    run_training.py. Consequences:

      * the package's own documented quickstart, `LeafletFA(adata, K=20); train()`, raises ValueError
        (waypoints_use defaults to True)
      * the Figure-2 benchmark's entire WAYPOINT ARM silently failed -- all 3 fits died with
        "PSI and PHI initializations for 5 waypoints not found", so the waypoints-vs-random
        comparison, one of the results the paper needs, produced nothing.

    Call this before `train()` when you want waypoint initialisation.

    Deliberately UNSTRATIFIED: waypoints are chosen by unsupervised max-min sampling over the PC
    space of junction usage ratios, with NO cell-type labels. The paper claims LeafletFA recovers
    splicing programs "without prior cell-type labels", so labels must not enter through the
    initialisation of the very model whose programs are then shown to recover cell types.
    """
    import numpy as np
    from sklearn.decomposition import TruncatedSVD

    # PCA space for waypoint selection
    if layer is not None:
        X = adata.layers[layer]
    elif "junc_ratio_centered" in adata.layers:
        X = adata.layers["junc_ratio_centered"]
    elif "junc_ratio" in adata.layers:
        X = adata.layers["junc_ratio"]
    else:
        X = adata.X
    # NB fit_transform ALREADY returns U*Sigma. Multiplying by singular_values_ again would give
    # U*Sigma^2 and collapse waypoint selection toward PC1.
    X_pca = TruncatedSVD(n_components=n_pca, random_state=seed).fit_transform(X)

    wp_idx = max_min_sampling(X_pca, K, num_components=n_dim, seed=seed)
    metacells = assign_nearest_cells(wp_idx, X_pca, num_nearest=metacell_size)

    rho = adata.layers["junc_ratio"] if "junc_ratio" in adata.layers else adata.X
    # Coverage mask: 1 where the junction's ATSE HAS READS, so an observed PSI of 0.0 counts as DATA
    # rather than being silently dropped as "missing".
    mask = None
    if mask_layer is not None and mask_layer in adata.layers:
        mask = adata.layers[mask_layer]
    elif "psi_mask" in adata.layers:
        mask = adata.layers["psi_mask"]
    elif "cell_by_cluster_matrix" in adata.layers:
        import scipy.sparse as sp
        C = adata.layers["cell_by_cluster_matrix"]
        mask = (C > 0) if sp.issparse(C) else (np.asarray(C) > 0)

    psi_init, phi_init = initialize_psi_phi(rho, metacells, mask=mask)

    # initialize_psi_phi returns psi as (J, K) -- which is ALREADY the varm layout (n_vars, K).
    # train() reads varm[key].T to get its (K, J). Do NOT transpose here.
    psi_init = np.asarray(psi_init, dtype=np.float32)      # (J, K)
    phi_init = np.asarray(phi_init, dtype=np.float32)      # (n_cells, K)
    assert psi_init.shape == (adata.n_vars, K), \
        f"psi_init is {psi_init.shape}, expected (n_vars={adata.n_vars}, K={K}) for varm"
    assert phi_init.shape == (adata.n_obs, K), \
        f"phi_init is {phi_init.shape}, expected (n_obs={adata.n_obs}, K={K}) for obsm"
    adata.varm[f"psi_init_{K}_waypoints"] = psi_init
    adata.obsm[f"phi_init_{K}_waypoints"] = phi_init
    return psi_init, phi_init
