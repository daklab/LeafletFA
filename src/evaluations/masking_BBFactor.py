# Core libraries
import os
import sys
import json
from datetime import datetime
import anndata as ad

# Scientific computing libraries
import numpy as np
import torch
import scipy.sparse as sp
import scipy.stats
from scipy.stats import binom

# Single-cell data manipulation
import anndata

# Pyro for probabilistic modeling
import pyro
import pyro.distributions as dist

# Machine learning libraries
from sklearn.decomposition import NMF, MiniBatchNMF
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    confusion_matrix, silhouette_score, roc_curve
)

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap.umap_ as umap

# Progress bar utility
from tqdm import tqdm

# Data handling
import pandas as pd

# Argument parsing
import argparse

# Import custom modules
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/beta-dirichlet-factor')
import factor_model

sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/clustering/')
import load_cluster_data as llc 

sys.path.append("/gpfs/commons/home/kisaev/Leaflet-private/src/simulation/")
import simulate_counts as sim 

sys.path.append("/gpfs/commons/home/kisaev/Leaflet-private/src/visualization/")
import vis as vis

sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/evaluations/')
import cost_correlation_assign
import differential_splicing
import masking_BBFactor as mask 
from scipy.sparse import csr_matrix

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run differential splicing analysis.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input .h5ad file.')
    parser.add_argument('--cell_type_column', type=str, default=None, help='Column name for cell types in the AnnData object.')
    parser.add_argument('--K_use', type=int, default=3, help='Number of clusters to use if cell_type_column is None.')
    parser.add_argument('--use_global_prior', action='store_true', help='Whether to use global prior in the factor model.')
    parser.add_argument('--input_conc_prior', type=str, default="inf", help='Input concentration prior value for the factor model.')
    parser.add_argument('--num_inits', type=int, default=3, help='Number of times to randomly initialize our factor model.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to run model training.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the model training.')
    parser.add_argument('--mask_perc', type=float, default=0.1, help='Percentage of data (nonzero cluster counts) to mask.')
    return parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

float_type = {"device": device, "dtype": torch.float}

if device == torch.device('cuda'):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

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
    num_nonzero = intron_clusts.count_nonzero()
    num_masked = int(num_nonzero * mask_percentage)

    # Ensure that mask_percentage is not too high
    assert num_masked < num_nonzero, "mask_percentage is too high."
    rows, cols = intron_clusts.nonzero()
    mask_indices = np.random.choice(len(rows), size=num_masked, replace=False)
    mask = np.zeros((num_cells, num_junctions), dtype=np.float32)
    mask[rows[mask_indices], cols[mask_indices]] = 1  # Mask selected indices
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
    indices = torch.tensor(nonzero_indices, dtype=torch.long)
    values = torch.tensor(masked_intron_clusts[nonzero_indices], dtype=torch.float)
    size = masked_intron_clusts.shape
    masked_intron_clusts_tensor = torch.sparse_coo_tensor(indices, values, size)
    
    # Use the same non-zero indices for the junction counts
    values_j = torch.tensor(masked_junction_counts[nonzero_indices], dtype=torch.float)
    masked_junction_counts_tensor = torch.sparse_coo_tensor(indices, values_j, size)

    # Ensure that the masked intron clusters are greater than or equal to the masked junction counts
    assert torch.all(masked_intron_clusts_tensor.to_dense() >= masked_junction_counts_tensor.to_dense()), \
        "Intron cluster counts must be >= junction counts."
    
    # Convert the masked matrices back to sparse CSR format for storage
    masked_intron_clusts_sparse = sp.csr_matrix(masked_intron_clusts)
    masked_junction_counts_sparse = sp.csr_matrix(masked_junction_counts)

    # Create a new AnnData object with masked data
    new_adata = anndata.AnnData(X=adata.X, obs=adata.obs, var=adata.var)

    # Save original unmasked values as well in the new AnnData object
    new_adata.layers["Original_Junction_Counts"] = sp.csr_matrix(junction_counts)
    new_adata.layers["Original_Cluster_Counts"] = sp.csr_matrix(intron_clusts)
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

def evaluate_model(adata, mask, model_psi, model_assign, bb_conc=None, true_juncs_layer="Original_Junction_Counts", true_clusts_layer="Original_Cluster_Counts"):
    '''
    Evaluate the factor model on masked data by comparing true and predicted PSI values.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with true junction and cluster counts.
    mask : torch.Tensor or np.ndarray
        Mask indicating the positions to evaluate (True for masked, False for unmasked).
    model_psi : np.ndarray
        The predicted PSI values from the model (latent variable `psis`).
    model_assign : np.ndarray
        The assignment probabilities for cells (latent variable `assign_post`).
    bb_conc : float, optional
        Concentration parameter for the Beta-Binomial distribution. If provided, log-likelihood will be calculated using Beta-Binomial.
    true_juncs_layer : str, optional
        Name of the layer in `adata` containing true junction counts.
    true_clusts_layer : str, optional
        Name of the layer in `adata` containing true cluster counts.

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
    
    # Ensure mask is cpu and is converted to boolean 
    mask_gen = np.array(mask.cpu(), dtype=bool)
    true_juncs = adata.layers[true_juncs_layer]
    true_clusts = adata.layers[true_clusts_layer]

    # Predicted PSI values: model_assign @ model_psi
    pred_psi = model_assign @ model_psi
    masked_pred = pred_psi[mask_gen]
    true_psi = true_juncs.toarray() / true_clusts.toarray()
    masked_true_psi = true_psi[mask_gen]

    # Ensure all true_clusts at masked indices are >= 1
    assert adata.layers[true_clusts_layer].toarray()[mask_gen].min() >= 1

    # Compute errors
    l1_error = np.mean(np.abs(masked_pred - masked_true_psi))
    l2_error = np.mean((masked_pred - masked_true_psi) ** 2)
    rmse = np.sqrt(l2_error)

    print(f"Number of values in masked_pred {len(masked_pred)}")
    print(f"Number of values in masked_true_psi {len(masked_true_psi)}")

    # Spearman correlation
    spearman_cor, _ = scipy.stats.spearmanr(masked_pred, masked_true_psi)

    # Log-likelihood (using Binomial or Beta-Binomial distribution)
    n_trials = true_clusts.toarray()[mask_gen]
    successes = true_juncs.toarray()[mask_gen]
    
    # Ensure bb_conc and masked_pred are both numpy arrays
    if isinstance(masked_pred, torch.Tensor):
        masked_pred = masked_pred.cpu().numpy()  # Convert to numpy if it's a torch tensor

    if bb_conc is not None:
        if torch.isinf(bb_conc).any():
            # If bb_conc is infinity, revert to Binomial distribution
            log_likelihood = np.sum(scipy.stats.binom.logpmf(successes, n_trials, masked_pred))
        else:
            # Beta-Binomial distribution
            if isinstance(bb_conc, torch.Tensor):
                bb_conc = bb_conc.cpu().numpy()  # Convert to numpy if it's a torch tensor
            alpha = masked_pred * bb_conc
            beta = (1 - masked_pred) * bb_conc
            log_likelihood = np.sum(scipy.stats.betabinom.logpmf(successes, n_trials, alpha, beta))
    else:
        # Binomial distribution
        log_likelihood = np.sum(scipy.stats.binom.logpmf(successes, n_trials, masked_pred))

    # Print results
    print(f"L1 error: {l1_error}\nSpearman correlation: {spearman_cor}\nL2 error: {l2_error}\nRMSE: {rmse}\nLog-likelihood: {log_likelihood}")

    return l1_error, spearman_cor, l2_error, rmse, log_likelihood

# NMF baseline model! 
def run_minibatch_nmf_baseline(adata, mask=None, n_components=10, batch_size=512, max_iter=200, random_state=42, true_juncs_layer="Original_Junction_Counts", true_clusts_layer="Original_Cluster_Counts"):
    '''
    Run NMF (Non-negative Matrix Factorization) on the PSI values derived from the masked AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with masked junction and cluster counts.
    mask : torch.Tensor or np.ndarray
        Mask indicating the positions to evaluate (True for masked, False for unmasked).
    n_components : int, optional
        Number of components for the NMF model.
    max_iter : int, optional
        Maximum number of iterations for the NMF solver.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    nmf_model : NMF
        The fitted NMF model.
    W : np.ndarray
        The NMF basis matrix (similar to assignment in the factor model).
    H : np.ndarray
        The NMF coefficient matrix (similar to psi in the factor model).
    '''

    # Extract junction counts and cluster counts
    junction_counts = adata.layers[true_juncs_layer].toarray()  # Convert sparse to dense if necessary
    cluster_counts = adata.layers[true_clusts_layer].toarray()
    
    # Compute PSI (junction counts / cluster counts), avoiding division by zero
    psi_matrix = np.divide(junction_counts, cluster_counts, out=np.zeros_like(junction_counts, dtype=float), where=(cluster_counts != 0))

    minibatch_nmf_model = MiniBatchNMF(n_components=n_components, batch_size=batch_size, max_iter=max_iter, random_state=random_state)

    if mask == None:
        print(f"No masking here!")
        # Fit NMF model to the imputed PSI matrix
        W = minibatch_nmf_model.fit_transform(psi_matrix)  # Basis matrix (similar to assignment)
        H = minibatch_nmf_model.components_  # Coefficient matrix (similar to psi)

    else: 
        # Mask to cpu 
        mask_gen = np.array(mask.cpu())
        # Apply mask (if necessary) - this depends on how you want to mask the data
        masked_psi = psi_matrix * (1 - mask_gen)
        # Fit NMF model to the imputed PSI matrix
        # Fit the MiniBatch NMF model (factorizes into W and H matrices)
        W = minibatch_nmf_model.fit_transform(masked_psi)  # Basis matrix (similar to assignment)
        H = minibatch_nmf_model.components_  # Coefficient matrix (similar to psi)  

    # Evaluate NMF predictions
    print(f"Finished fitting NMF with {n_components} components.")
    return W, H

def prepare_output_directory(mask_perc, K_use, use_global_prior, input_conc, num_inits, num_epochs, cell_type_column):
    """Prepares the output directory with timestamp and relevant parameters."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cell_type_str = f"CellType_{cell_type_column}" if cell_type_column else "NoCellType"
    use_global_prior_str = "GlobalPrior" if use_global_prior else "NoGlobalPrior"
    
    # Handle input_conc for directory naming
    input_conc_prior_str = "LearnedConc" if input_conc is None else f"ConcPrior_{input_conc}"
    
    output_dir = (f"./analysis_{timestamp}_MaskPerc_{mask_perc}_K_{K_use}_"
                  f"Prior_{use_global_prior_str}_NumEpochs_{num_epochs}_"
                  f"{input_conc_prior_str}_Inits_{num_inits}_"
                  f"{cell_type_str}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_parameters(output_dir, args, K, input_conc):
    """Saves run parameters as a JSON file."""
    parameters = {
        "mask_perc": args.mask_perc,
        "cell_type_column": args.cell_type_column,
        "K_use": K,
        "input_path": args.input_path,
        "use_global_prior": args.use_global_prior,
        "input_conc_prior": input_conc.item() if torch.is_tensor(input_conc) else input_conc,
        "num_inits": args.num_inits,
        "num_epochs": args.num_epochs,
        "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    json_path = os.path.join(output_dir, 'parameters.json')
    with open(json_path, 'w') as json_file:
        json.dump(parameters, json_file, indent=4)
    print(f"Parameters saved to {json_path}")

def handle_input_conc_prior(input_conc_prior):
    """Handles the conversion of input concentration prior to the appropriate format."""
    if input_conc_prior == "None":
        # If input_conc_prior is None, return None (or a default value if needed)
        return None
    elif isinstance(input_conc_prior, str):
        if input_conc_prior.lower() == "inf":
            return torch.tensor(np.inf, **float_type)
        else:
            return torch.tensor(float(input_conc_prior), **float_type)
    elif isinstance(input_conc_prior, float):
        return torch.tensor(np.inf, **float_type) if input_conc_prior == float('inf') else torch.tensor(input_conc_prior, **float_type)
    elif isinstance(input_conc_prior, torch.Tensor):
        return input_conc_prior.to(**float_type)
    else:
        raise ValueError("Unsupported type for input_conc_prior")

def run_factor_model(my_data, K, device, use_global_prior, input_conc, num_inits=3, num_epochs=100, lr=0.1):
    """Runs the factor model with specified parameters, including the learning rate."""
    
    # Convert sparse matrices to COO format, ensure they are on the GPU, and follow the specified float type
    full_y_tensor = my_data.ycount_lookup.to_sparse_coo()
    full_total_counts_tensor = my_data.tcount_lookup.to_sparse_coo()
    
    print(f"Running on device: {device}")
    print(f"use_global_prior: {use_global_prior}") 
    
    all_results = []
    all_params = []  # Store parameters for each initialization

    for _ in range(num_inits):
        pyro.clear_param_store()  # Clear the parameter store for each initialization
        result, variable_sizes = factor_model.main(
            full_y_tensor, 
            full_total_counts_tensor, 
            use_global_prior=use_global_prior, 
            num_initializations=1,
            K=K, 
            lr=lr, 
            input_conc_prior=input_conc, 
            loss_plot=False, 
            num_epochs=num_epochs, 
            save_to_file=False
        )

        all_results.append(result)
        
        # Store the parameters from the ParamStoreDict
        params_copy = {name: pyro.get_param_store().get_param(name).detach().clone()
                       for name in pyro.get_param_store().get_all_param_names()}
        all_params.append(params_copy)
    return all_results, all_params

def load_adata(input_path):
    """Loads an AnnData object from the specified path."""
    return ad.read_h5ad(input_path)

def main():

    """Main function to handle the analysis pipeline."""
    # Parse command-line arguments
    args = parse_arguments()

    # Convert input concentration prior into the appropriate type
    input_conc = handle_input_conc_prior(args.input_conc_prior)

    # Handle cell_type_column input
    cell_type_column = args.cell_type_column if args.cell_type_column != "None" else None

    # Extract parameters from arguments
    K_use = args.K_use
    input_path = args.input_path
    use_global_prior = args.use_global_prior
    num_inits = args.num_inits
    num_epochs = args.num_epochs
    lr = args.lr  # Learning rate
    mask_perc = args.mask_perc

    # Prepare the output directory with timestamp
    output_dir = prepare_output_directory(mask_perc, K_use, use_global_prior, input_conc, num_inits, num_epochs, cell_type_column)

    # Load data and set K
    adata = load_adata(input_path)
    K = adata.obs[cell_type_column].nunique() if cell_type_column else K_use

    # Save run parameters as JSON
    save_parameters(output_dir, args, K, input_conc)

    # Generate mask for the data 
    mask_gen, seed = generate_mask(adata, mask_percentage=mask_perc)

    # Apply mask 
    new_adata, my_data = apply_mask_to_anndata(adata, mask_gen, cluster_layer="Cluster_Counts", junction_layer="Junction_Counts")

    # Run factor model!
    all_results, all_params = run_factor_model(my_data, K, device, use_global_prior, input_conc, num_inits, num_epochs, lr)
    
    # Extract Latent Variables 
    # Select the best initialization based on the loss
    best_init = np.argmin([result[0]["losses"][-1] for result in all_results])
    latent_vars = all_results[best_init][0]['summary_stats']
    
    # Differential Splicing Analysis
    assign_post = latent_vars["assign"]["mean"]
    model_psi = latent_vars["psi"]["mean"]

    # If input_conc was None extract it from latent variables 
    if input_conc == None:
        bb_conc = latent_vars["bb_conc"]["mean"]
    else:
        bb_conc = None

    # Evaluate learned parameters on masked data 
    l1_error, spearman_cor, l2_error, rmse, log_likelihood = evaluate_model(
        new_adata, mask_gen, model_psi, assign_post, bb_conc=bb_conc, 
        true_juncs_layer="Original_Junction_Counts", 
        true_clusts_layer="Original_Cluster_Counts"
    )
    
    # Run NMF 
    W, H = run_minibatch_nmf_baseline(
        new_adata, mask=mask_gen, n_components=K, batch_size=512, max_iter=200, 
        random_state=42, true_juncs_layer="Original_Junction_Counts", 
        true_clusts_layer="Original_Cluster_Counts"
    )

    # Evaluate NMF results 
    l1_error_NMF, spearman_cor_NMF, l2_error_NMF, rmse_NMF = evaluate_model(
        new_adata, mask_gen, H, W, 
        true_juncs_layer="Original_Junction_Counts", 
        true_clusts_layer="Original_Cluster_Counts"
    )

    # Prepare results for saving
    factor_model_results = {
        "Model": "FactorModel",
        "L1 Error": l1_error,
        "Spearman Correlation": spearman_cor,
        "L2 Error": l2_error,
        "RMSE": rmse,
        "Log Likelihood": log_likelihood
    }
    
    nmf_model_results = {
        "Model": "NMF",
        "L1 Error": l1_error_NMF,
        "Spearman Correlation": spearman_cor_NMF,
        "L2 Error": l2_error_NMF,
        "RMSE": rmse_NMF, 
        "Log Likelihood": None
    }

    # Combine results into a DataFrame
    results_df = pd.DataFrame([factor_model_results, nmf_model_results])

    # Save results to a CSV file in the output directory
    results_path = os.path.join(output_dir, "model_comparison_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()