import numpy as np
import pyro
import pyro.distributions as dist
import torch
import scipy.sparse as sp
import scipy 
import scipy.stats
from scipy.stats import binom
import scipy.stats
import numpy as np
import torch
import scipy.sparse as sp
import scipy.stats
import anndata
import sys

sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/clustering/')
import load_cluster_data as llc 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

float_type = {"device": device, "dtype": torch.float}
if device == torch.device('cuda'):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Cluster + junction counts set to zero for those predefined indices 
# From model fit, get estimate of what those values should be given learned PSI values 
# Use Likelihood for test elements where might put more weight on juntion with higher counts 
# L1 mean absolute error can be used to evaluate imputed vs observed PSI values for J-C pairs

def generate_mask(adata, layer_key="Cluster_Counts", mask_percentage=0.1, seed=42, randomize_seed=False):
    '''Generate a mask for the specified layer of an AnnData object.'''

    # Set seed for reproducibility
    seed = np.random.randint(0, 1000000) if randomize_seed else seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Extract the intron cluster layer (must be sparse)
    intron_clusts = adata.layers[layer_key]
    if not sp.issparse(intron_clusts):
        raise ValueError(f"{layer_key} must be a sparse matrix.")

    # Get number of cells and junctions
    num_cells, num_junctions = intron_clusts.shape

    # Get the number of non-zero entries and calculate how many to mask
    num_nonzero = intron_clusts.count_nonzero()
    num_masked = int(num_nonzero * mask_percentage)

    # Ensure that mask_percentage is not too high
    assert num_masked < num_nonzero, "mask_percentage is too high."

    # Get non-zero indices (rows, cols) where values are >=1 in intron_clusts
    rows, cols = intron_clusts.nonzero()

    # Sample a subset of non-zero indices to mask
    mask_indices = np.random.choice(len(rows), size=num_masked, replace=False)

    # Create a mask with the same shape as the original matrix
    mask = np.zeros((num_cells, num_junctions), dtype=np.float32)
    mask[rows[mask_indices], cols[mask_indices]] = 1  # Mask selected indices

    # Convert the mask to a torch tensor
    mask_tensor = torch.tensor(mask, dtype=torch.float32)

    print(f"Total masked entries: {np.sum(mask)}")
    return mask_tensor, seed

def apply_mask_to_anndata(adata, mask, cluster_layer="Cluster_Counts", junction_layer="Junction_Counts"):
    '''
    Apply a mask to the intron cluster matrix and junction count matrix, 
    and return a new AnnData object with masked entries.
    '''

    # Ensure the mask is a NumPy array for element-wise operations
    mask = mask.cpu().numpy()

    # Extract the intron cluster and junction counts layers
    intron_clusts = adata.layers[cluster_layer].toarray() if sp.issparse(adata.layers[cluster_layer]) else adata.layers[cluster_layer]
    junction_counts = adata.layers[junction_layer].toarray() if sp.issparse(adata.layers[junction_layer]) else adata.layers[junction_layer]

    # Apply the mask (1 means masked, 0 means untouched)
    masked_intron_clusts = intron_clusts * (1 - mask)
    masked_junction_counts = junction_counts * (1 - mask)

    # Get the non-zero indices from the masked intron clusters
    nonzero_indices = np.nonzero(masked_intron_clusts)

    # Convert intron clusters to sparse tensor
    indices = torch.tensor(nonzero_indices, dtype=torch.long)
    values = torch.tensor(masked_intron_clusts[nonzero_indices], dtype=torch.float)
    size = masked_intron_clusts.shape
    masked_intron_clusts_tensor = torch.sparse_coo_tensor(indices, values, size)
    
    # Use the same non-zero indices for the junction counts
    values_j = torch.tensor(masked_junction_counts[nonzero_indices], dtype=torch.float)
    masked_junction_counts_tensor = torch.sparse_coo_tensor(indices, values_j, size)

    # Ensure that the masked intron clusters are greater than or equal to the masked junction counts
    assert torch.all(masked_intron_clusts_tensor.to_dense() >= masked_junction_counts_tensor.to_dense())

    # Convert the masked matrices back to sparse CSR format for storage
    masked_intron_clusts_sparse = sp.csr_matrix(masked_intron_clusts)
    masked_junction_counts_sparse = sp.csr_matrix(masked_junction_counts)

    # Print the number of non-zero elements for validation
    print(f"Masked_Cluster_Counts nnz: {masked_intron_clusts_sparse.nnz}")
    print(f"Masked_Junction_Counts nnz: {masked_junction_counts_sparse.nnz}")

    # Create a new AnnData object with masked data
    new_adata = anndata.AnnData(X=adata.X, obs=adata.obs, var=adata.var)
    new_adata.layers["Masked_Cluster_Counts"] = masked_intron_clusts_sparse.tocoo()
    new_adata.layers["Masked_Junction_Counts"] = masked_junction_counts_sparse.tocoo()

    # Convert back to sparse tensors for model input
    cell_index_tensor, junc_index_tensor, my_data = llc.make_torch_adata(
        new_adata, 
        cluster_layer="Masked_Cluster_Counts", 
        junction_layer="Masked_Junction_Counts", 
        **float_type
    )

    return new_adata, my_data

def evaluate_model(true_juncs, true_clusts, model_psi, model_assign, mask):
    '''
    Evaluate the factor model on masked data by comparing true and predicted PSI values.

    Parameters
    ----------
    true_juncs : torch.Tensor
        True unmasked junction counts.
    true_clusts : torch.Tensor
        True unmasked intron clusters.
    model_psi : torch.Tensor
        Cell-specific factor loadings (J x K matrix).
    model_assign : torch.Tensor
        Cell-specific factor assignments (C x K matrix).
    mask : numpy.ndarray
        Binary mask indicating the entries to evaluate.

    Returns
    -------
    l1_error : float
        Mean absolute error between masked predicted and true PSI values.
    spearman_cor : float
        Spearman correlation between masked predicted and true PSI values.
    l2_error : float
        Mean squared error between masked predicted and true PSI values.
    rmse : float
        Root mean squared error.
    log_likelihood : float
        Log-likelihood under a Beta-Binomial distribution.
    '''
    # Predicted PSI values: model_assign @ model_psi
    pred_psi = model_assign @ model_psi

    # Masked entries
    masked_pred = pred_psi[mask]
    true_psi = true_juncs / true_clusts
    masked_true_psi = true_psi[mask]

    # Ensure all true_clusts at masked indices are >= 1
    assert true_clusts[mask].min() >= 1

    # Compute errors
    l1_error = np.mean(np.abs(masked_pred - masked_true_psi))
    l2_error = np.mean((masked_pred - masked_true_psi) ** 2)
    rmse = np.sqrt(l2_error)

    # Spearman correlation
    spearman_cor, _ = scipy.stats.spearmanr(masked_pred, masked_true_psi)

    # Log-likelihood (using Binomial distribution)
    n_trials = true_clusts[mask]
    successes = true_juncs[mask]
    log_likelihood = np.sum(scipy.stats.binom.logpmf(successes, n_trials, masked_pred))

    # Print results
    print(f"L1 error: {l1_error}\nSpearman correlation: {spearman_cor}\nL2 error: {l2_error}\nRMSE: {rmse}\nLog-likelihood: {log_likelihood}")

    return l1_error, spearman_cor, l2_error, rmse, log_likelihood
