import argparse
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run differential splicing analysis.')

    # Required arguments
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input .h5ad file.')
    parser.add_argument('--ATSE_file', type=str, help='File containing ATSE annotations from the previous Leaflet processing step.')

    # Optional arguments with default values
    parser.add_argument('--cell_type_column', type=str, default=None, help='Column name for cell types in the AnnData object.')
    parser.add_argument('--K_use', type=int, default=3, help='Number of clusters to use if cell_type_column is None.')
    parser.add_argument('--input_conc_prior', type=str, default="inf", help='Input concentration prior value for the factor model.')
    parser.add_argument('--num_inits', type=int, default=3, help='Number of times to randomly initialize our factor model.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to run model training.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the model training.')

    # Boolean flags (default False, set to True when provided)
    parser.add_argument('--use_global_prior', action='store_true', help='Whether to use junctin specific priors in the factor model (default: False).')
    parser.add_argument('--waypoints_use', action='store_true', help="Indicate whether the factor model should be initialized with predefined matrices (default: False).")
    parser.add_argument('--brain_only', action='store_true', help="Indicate whether the model will be run only on brain data (default: False).")
    parser.add_argument('--run_NMF', action='store_true', help='Indicate whether NMF should be run as a baseline (default: False).')
    parser.add_argument('--mask_perc', type=float, default=0.1, help='Percentage of data (nonzero cluster counts) to mask.')

    return parser.parse_args()


import sys
import os
import json
import numpy as np
import torch
import anndata as ad
from importlib import reload
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import pyro 
import umap.umap_ as umap
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import entropy
# Machine learning libraries
from sklearn.decomposition import NMF, MiniBatchNMF
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    confusion_matrix, silhouette_score, roc_curve
)

# Import custom modules
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/beta-dirichlet-factor')
import factor_model
reload(factor_model)

sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/evaluations/')
import cost_correlation_assign
import differential_splicing
import masking_BBFactor as mask 

import scipy.sparse as sp 

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
    
    #np.random.seed(seed)
    
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
    new_adata = ad.AnnData(X=adata.X, obs=adata.obs, var=adata.var)

    # Save original unmasked values as well in the new AnnData object
    new_adata.layers["Original_Junction_Counts"] = sp.csr_matrix(junction_counts)
    new_adata.layers["Original_Cluster_Counts"] = sp.csr_matrix(intron_clusts)
    new_adata.layers["Masked_Cluster_Counts"] = masked_intron_clusts_sparse.tocoo()
    new_adata.layers["Masked_Junction_Counts"] = masked_junction_counts_sparse.tocoo()

    # Convert sparse matrices to COO format
    full_y_tensor = masked_junction_counts_tensor
    full_total_counts_tensor = masked_intron_clusts_tensor

    return new_adata, full_y_tensor, full_total_counts_tensor

# adata_input.layers["cell_by_cluster_matrix"].toarray()[0, 186:189]

def evaluate_model(adata, mask, model_psi, model_assign, bb_conc=None, NMF=False, true_juncs_layer="Original_Junction_Counts", true_clusts_layer="Original_Cluster_Counts"):
    '''
    Evaluate the factor model on masked data by comparing true and predicted PSI values.

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

    if not NMF: 
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
        return l1_error, spearman_cor, l2_error, rmse, log_likelihood
    
    return l1_error, spearman_cor, l2_error, rmse

def prepare_output_directory(mask_perc, K_use, use_global_prior, input_conc, num_inits, num_epochs, cell_type_column, waypoint):
    """Prepares the output directory with timestamp and relevant parameters."""

    # Generate a random number to ensure uniqueness (e.g., 5-digit random integer)
    random_number = np.random.randint(10000, 99999)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cell_type_str = f"CellType_{cell_type_column}" if cell_type_column else "NoCellType"
    use_global_prior_str = "GlobalPrior" if use_global_prior else "NoGlobalPrior"
    waypoint_str = "UsingWaypoints" if waypoint else "NoWaypoints"
    
    # Handle input_conc for directory naming
    input_conc_prior_str = "LearnedConc" if input_conc is None else f"ConcPrior_{input_conc}"
    
    # Construct the output directory name with random number appended
    output_dir = (f"./analysis_{timestamp}_MaskPerc_{mask_perc}_K_{K_use}_{waypoint_str}_"
                  f"Prior_{use_global_prior_str}_NumEpochs_{num_epochs}_"
                  f"{input_conc_prior_str}_Inits_{num_inits}_"
                  f"{cell_type_str}_Random_{random_number}")
    
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

def run_factor_model(adata_input, full_y_tensor, full_total_counts_tensor, K, device, use_global_prior, input_conc, waypoints_use, output_dir, num_inits=3, num_epochs=100, lr=0.1):
    """Runs the factor model with specified parameters, including the learning rate."""
    
    print(f"Running on device: {device}")
    print(f"use_global_prior: {use_global_prior}") 
    
    all_results = []
    all_params = []  # Store parameters for each initialization

    # No waypoint initialization in this analysis 
    psi_init = None 
    phi_init = None
    
    for _ in range(num_inits):
        # Clear the parameter store for each initialization
        pyro.clear_param_store()  

        # Run the factor model
        result, variable_sizes = factor_model.main(
            full_y_tensor, 
            full_total_counts_tensor, 
            psi_init= psi_init, 
            phi_init= phi_init, 
            use_global_prior=use_global_prior, 
            num_initializations=1,
            K=K, 
            lr=lr, 
            input_conc_prior=input_conc, 
            loss_plot=True, 
            num_epochs=num_epochs, 
            save_to_file=False, 
            output_dir=output_dir
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
    ATSE_file = args.ATSE_file 
    run_NMF = args.run_NMF
    waypoints_use = args.waypoints_use
    brain_only = args.brain_only
    mask_perc = args.mask_perc

    # Read in the intron cluster file 
    print("Reading in obtained intron cluster (ATSE file!)")
    intron_clusts = pd.read_csv(ATSE_file, sep="}")
    genes = intron_clusts[["gene_id", "gene_name"]].drop_duplicates()

    if brain_only:
        print(f"Running model only on brain cells!")
    else:
        print(f"Running on all tissues!")
        
    # Prepare the output directory with timestamp
    output_dir = prepare_output_directory(mask_perc, K_use, use_global_prior, input_conc, num_inits, num_epochs, cell_type_column, waypoints_use)

    # Load data and set K
    adata = load_adata(input_path)
    adata.var = pd.merge(adata.var, genes[['gene_id', 'gene_name']], how='left', on='gene_id')

    # K = adata.obs[cell_type_column].nunique() if cell_type_column else K_use
    K = K_use
    print(f"K going into model training is: {K}!")

    # Save run parameters as JSON
    save_parameters(output_dir, args, K, input_conc)

    # Define the path to save/load tensor files
    path_tosaveto = '/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/'

    # Determine which file names to use based on brain_only flag
    if brain_only:
        full_y_tensor_filename = 'full_y_tensor_brain_only.pt'
        full_total_counts_tensor_filename = 'full_total_counts_tensor_brain_only.pt'
    else:
        full_y_tensor_filename = 'full_y_tensor.pt'
        full_total_counts_tensor_filename = 'full_total_counts_tensor.pt'

    print(f"Using {full_y_tensor_filename} and {full_total_counts_tensor_filename} !")

    # Full paths to the tensor files
    full_y_tensor_path = os.path.join(path_tosaveto, full_y_tensor_filename)
    full_total_counts_tensor_path = os.path.join(path_tosaveto, full_total_counts_tensor_filename)

    # Load the sparse tensors first on CPU
    full_y_tensor = torch.load(full_y_tensor_path, map_location='cpu')
    full_total_counts_tensor = torch.load(full_total_counts_tensor_path, map_location='cpu')
    
    # Move to gpu if using 
    full_y_tensor = full_y_tensor.to(device)
    full_total_counts_tensor = full_total_counts_tensor.to(device)

    # Generate mask for the data 
    mask_gen, seed = generate_mask(adata, layer_key="cell_by_cluster_matrix", mask_percentage=mask_perc)

    # Apply mask 
    new_adata, new_full_y_tensor, new_full_total_counts_tensor = apply_mask_to_anndata(adata, mask_gen, cluster_layer="cell_by_cluster_matrix", junction_layer="cell_by_junction_matrix")

    # Run factor model!   
    all_results, all_params = run_factor_model(new_adata, new_full_y_tensor, new_full_total_counts_tensor, K, device, use_global_prior, input_conc, waypoints_use, output_dir, num_inits, num_epochs, lr)

    # Extract Latent Variables 
    # Select the best initialization based on the loss
    best_init = np.argmin([result[0]["losses"][-1] for result in all_results])
    latent_vars = all_results[best_init][0]['summary_stats']
    best_elbo = all_results[best_init][0]["losses"][-1]

    # Differential Splicing Analysis
    assign_post = latent_vars["assign"]["mean"]
    model_psi = latent_vars["psi"]["mean"]
    input_conc_provided = input_conc

    if input_conc is None:
        bb_conc = torch.tensor(latent_vars["bb_conc"]["mean"])
    else:
        bb_conc = None

    print(f"bb_conc is: {bb_conc}!")

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
        new_adata, mask_gen, H, W, NMF=True, 
        true_juncs_layer="Original_Junction_Counts", 
        true_clusts_layer="Original_Cluster_Counts"
    )

    # NMF Model results with additional fields set to None/NaN to match FactorModel
    nmf_model_results = {
        "Model": "NMF",
        "K": K_use,
        "mask_perc": mask_perc,
        "use_global_prior": None,
        "lr": None,
        "input_conc": None,
        "input_conc_provided": None,
        "best_elbo": None,
        "L1 Error": l1_error_NMF,
        "Spearman Correlation": spearman_cor_NMF,
        "L2 Error": l2_error_NMF,
        "RMSE": rmse_NMF, 
        "Log Likelihood": None
    }


    # Prepare results for saving
    factor_model_results = {
        "Model": "FactorModel",
        "K": K_use, 
        "mask_perc": mask_perc, 
        "use_global_prior": use_global_prior, 
        "lr": lr,
        "input_conc": input_conc, 
        "input_conc_provided": input_conc_provided, 
        "best_elbo": best_elbo,
        "L1 Error": l1_error,
        "Spearman Correlation": spearman_cor,
        "L2 Error": l2_error,
        "RMSE": rmse,
        "Log Likelihood": log_likelihood
    }
    
    # Combine results into a DataFrame
    results_df = pd.DataFrame([factor_model_results, nmf_model_results])

    # Save results to a CSV file in the output directory
    results_path = os.path.join(output_dir, "model_comparison_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()