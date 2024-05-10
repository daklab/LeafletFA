import numpy as np
import pyro
import pyro.distributions as dist
import torch
import scipy.sparse as sp
import scipy 
import scipy.stats
from scipy.stats import binom

# Cluster + junction counts set to zero for those predefined indices 
# From model fit, get estimate of what those values should be given learned PSI values 
# Use Likelihood for test elements where might put more weight on juntion with higher counts 
# L1 mean absolute error can be used to evaluate imputed vs observed PSI values for J-C pairs


def generate_mask(intron_clusts, mask_percentage=0.1, seed=42, randomize_seed=False):

    '''
    Generate a mask for a given intron cluster matrix.

    Parameters
    ----------
    intron_clusts : 
        A C x J sparse matrix of type '<class 'numpy.int64'>'
	with stored elements in COOrdinate format> [intron cluster counts]

    mask_percentage : float
        The percentage of entries to mask. Default is 0.1.

    Returns
    -------
    mask : torch.Tensor
        A C x J matrix of 0s and 1s where 1s indicate masked entries.
    '''

    # Set seed
    if randomize_seed:
        seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        np.random.seed(seed)
        torch.manual_seed(seed)

    print("The seed is: ", seed)

    # Get number of cells and junctions
    num_cells = intron_clusts.shape[0]
    num_junctions = intron_clusts.shape[1]

    # Get number of entries to mask
    num_masked = int(intron_clusts.nnz * mask_percentage)

    # Make sure this number is less than total number of non-zero entries in intron_clusts
    assert num_masked < intron_clusts.nnz

    # Get indices of entries to mask, only from indices where values are >=1 in intron_clusts
    indices_to_mask_from = np.nonzero(intron_clusts)
    # Assert intron_clusts at these indices is >=1
    assert np.all(intron_clusts.toarray()[indices_to_mask_from] >= 1)

    # sample the mask_percentage amount of indices_to_mask_from 
    indices = np.random.choice(len(indices_to_mask_from[0]), size=num_masked, replace=False)
    print(indices[0:50])  

    # Getting pairs
    mask_rows_ind = indices_to_mask_from[0]
    mask_cols_ind = indices_to_mask_from[1]

    # Create mask
    mask = np.zeros((num_cells, num_junctions))
    mask[mask_rows_ind[indices], mask_cols_ind[indices]] = 1

    # check values of intron_clusts at mask == 1
    assert np.all(intron_clusts.toarray()[mask == 1] >= 1)

    print("Number of entries (junction-cell pairs) masked: ", np.sum(mask))
    return mask, seed

# Second function to apply mask to intron cluster matrix and junction count matrix

def apply_mask(junction_counts, intron_clusts, mask):
    
        '''
        Apply a mask to an intron cluster matrix and junction count matrix.
    
        Parameters
        ----------
        intron_clusts : 
        A C x J sparse matrix of type '<class 'numpy.int64'>'
	    with stored elements in COOrdinate format> [intron cluster counts]
    
        intron_clusts : 
        A C x J sparse matrix of type '<class 'numpy.int64'>'
	    with stored elements in COOrdinate format> [intron cluster counts]
    
        mask : torch.Tensor
            A C x J matrix of 0s and 1s where 1s indicate masked entries.
    
        Returns
        -------
        masked_intron_clusts : torch.Tensor
            A C x J matrix of intron clusters with masked entries set to 0.
    
        masked_junction_counts : torch.Tensor
            A C x J matrix of junction counts with masked entries set to 0.
        '''
        
        masked_intron_clusts = intron_clusts.toarray() * (1 - mask)
    
        # Mask junction counts
        masked_junction_counts = junction_counts.toarray() * (1 - mask)
    
        return masked_junction_counts, masked_intron_clusts

def prep_model_input(masked_junction_counts, masked_intron_clusts):
   
    '''
    Prepare input for factor model.

    Parameters
    ----------
    masked_junction_counts : torch.Tensor
        A C x J matrix of junction counts with masked entries set to 0.

    masked_intron_clusts : torch.Tensor
        A C x J matrix of intron clusters with masked entries set to 0.
    
    Returns
    -------
    masked_junction_counts_tensor : torch.sparse_coo_tensor
        A sparse tensor of masked junction counts.
    
    masked_intron_clusts_tensor : torch.sparse_coo_tensor
        A sparse tensor of masked intron clusters.
    ''' 

    # First make intron cluster sparse tensor 

    #1. intron clusts 
    indices = torch.tensor(np.nonzero(masked_intron_clusts), dtype=torch.long)
    values = torch.tensor(masked_intron_clusts[np.nonzero(masked_intron_clusts)], dtype=torch.float)
    # Determine the size of the tensor
    num_cells = masked_intron_clusts.shape[0]
    num_junctions = masked_intron_clusts.shape[1]
    size = (num_cells, num_junctions)
    # Create a sparse tensor
    masked_intron_clusts_tensor = torch.sparse_coo_tensor(indices, values, size)
    masked_intron_clusts_tensor

    #2. use the same indices to make a sparse tensor from masked_junction_counts
    values_j = torch.tensor(masked_junction_counts[np.nonzero(masked_intron_clusts)], dtype=torch.float)
    # Keep same size tensor as introns 
    masked_junction_counts_tensor = torch.sparse_coo_tensor(indices, values_j, size)

    assert torch.all(masked_intron_clusts_tensor.to_dense() >= masked_junction_counts_tensor.to_dense())

    return masked_junction_counts_tensor, masked_intron_clusts_tensor


# need a slightly different function for evaluating mixture model 

def evaluate_mixture_model(true_juncs, true_clusts, model_res, masked_matrix):
    '''
    Evaluate trained mixture model on masked data.

    Parameters:
    - true_juncs (torch.Tensor): True junction counts.
    - true_clusts (torch.Tensor): True cluster counts.
    - model_res (tuple): Tuple containing the results of the trained mixture model, 
                         including ALPHA_f, PI_f, GAMMA_f, PHI_f, and elbos_all.
    - masked_matrix (numpy.ndarray): Masked matrix indicating the entries to evaluate.

    Returns:
    - l1_error (torch.Tensor): L1 error (mean absolute difference) between predicted and true PSI values.
    - l2_error (torch.Tensor): L2 error (root mean squared error) between predicted and true PSI values.
    - correlation_coefficient (float): Pearson correlation coefficient between predicted and true PSI values.
    - r_squared (torch.Tensor): R-squared (coefficient of determination) between predicted and true PSI values.
    - log_likelihood (float): Log-likelihood score for how well the data fits using the predicted PSI values.
    '''

    # Extract latent variables from trained model 
    ALPHA_f, PI_f, GAMMA_f, PHI_f, elbos_all = model_res
    psi = ALPHA_f / (ALPHA_f+PI_f)   

    # Calculate predicted PSI values for each cell and junction
    pred = PHI_f @ psi.T 

    # Let's look at only the masked entries
    masked_pred = pred[np.nonzero(masked_matrix)]
    true_psi = true_juncs / true_clusts

    true_clusts_dense = true_clusts.toarray()
    junc_counts_dense = true_juncs.toarray()

    # Get true_psi values for masked indices 
    masked_true_psi = true_psi[np.nonzero(masked_matrix)]

    # Calculate L1 error
    l1_error = torch.mean(torch.abs(masked_pred - masked_true_psi))

    # Calculate L2 error (Root Mean Squared Error)
    l2_error = torch.sqrt(torch.mean((masked_pred - masked_true_psi) ** 2))

    # Calculate correlation coefficient
    correlation_coefficient = np.corrcoef(masked_pred, masked_true_psi)[0, 1]

    # Calculate log-likelihood
    n_trials = true_clusts_dense[np.nonzero(masked_matrix)]
    successes = junc_counts_dense[np.nonzero(masked_matrix)]
    log_likelihood = np.sum(binom.logpmf(successes, n_trials, masked_pred))

    return l1_error, l2_error, correlation_coefficient, log_likelihood


# next function shoould evaluate model fit on masked data

def evaluate_model(true_juncs, true_clusts, model_psi, model_assign, mask):
    
    '''
    Evaluate the factor model on masked data.

    Parameters
    ----------
    true_juncs : torch.Tensor
        A C x J matrix of true unmasked junction counts.

    true_clusts : torch.Tensor          
        A C x J matrix of true unmasked intron clusters. 

    model_psi : torch.Tensor
        A J x K matrix of cell-specific factor loadings.

    model_assign : torch.Tensor
        A C x K matrix of cell-specific factor assignments.

    mask : numpy.ndarray
        A binary mask indicating the entries to evaluate.

    Returns
    -------
    l1_error : float
        The mean absolute difference between masked predicted and true PSI values.

    spearman_cor : float
        The Spearman correlation coefficient between masked predicted and true PSI values.

    l2_error : float
        The mean squared error between masked predicted and true PSI values.

    rmse : float
        The root mean squared error between masked predicted and true PSI values.

    log_likelihood : float
        The log-likelihood of the observed data given the predicted PSI values, assuming a Beta-Binomial distribution.
    '''

    # get predicted PSI values for each cell and junction
    pred = model_assign @ model_psi # predicted PSI values for each cell and junction

    # let's look at only the masked entries
    masked_pred = pred[np.nonzero(mask)]
    true_psi = true_juncs / true_clusts

    # assert that true cluster counts at masked indices were at least 1 
    assert true_clusts[np.nonzero(mask)].min() >=1 

    # get true_psi values for masked indices 
    masked_true_psi = true_psi[np.nonzero(mask)]

    # get L1 absolute mean error between masked predicted and true PSI values
    l1_error = np.mean(np.abs(masked_pred - masked_true_psi))

    # get another measure of error between predicted and true PSI values
    l2_error = np.mean((masked_pred - masked_true_psi)**2)

    # report root mean square error instead of l2 (more like std dev which would be more intuitive)
    rmse = np.sqrt(l2_error)

    # get spearman correlation between masked predicted and true PSI values
    spearman_cor = scipy.stats.spearmanr(masked_pred, masked_true_psi)[0]

    # Calculate log-likelihood for Beta-Binomial distribution
    n_trials = true_clusts[np.nonzero(mask)]
    successes = true_juncs[np.nonzero(mask)]
    log_likelihood = np.sum(scipy.stats.binom.logpmf(successes, n_trials, masked_pred))

    print("L1 error: ", l1_error)
    print("Spearman correlation: ", spearman_cor)
    print("L2 error: ", l2_error)
    print("RMSE: ", rmse)
    print("Log-likelihood: ", log_likelihood)

    return l1_error, spearman_cor, l2_error, rmse, log_likelihood