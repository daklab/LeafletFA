import argparse
import pickle

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
from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import scipy.sparse
from scipy.sparse import csr_matrix

# Import custom modules
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/beta-dirichlet-factor')
import factor_model
reload(factor_model)

sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/clustering/')
import load_cluster_data as llc 

sys.path.append("/gpfs/commons/home/kisaev/Leaflet-private/src/simulation/")
import simulate_counts as sim 
reload(sim)

sys.path.append("/gpfs/commons/home/kisaev/Leaflet-private/src/visualization/")
import vis as vis

sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/evaluations/')
import cost_correlation_assign
import differential_splicing
import masking_BBFactor as mask 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
float_type = {"device": device, "dtype": torch.float}
if device == torch.device('cuda'):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def create_report_file(output_dir):
    """Creates a report file for logging results."""
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    return open(report_path, 'w')

def log_to_report(report_file, text):
    """Logs messages to the report file and prints to the console."""
    report_file.write(text + '\n')
    print(text)

def load_adata(input_path):
    """Loads an AnnData object from the specified path."""
    return ad.read_h5ad(input_path)

def average_pairwise_correlation(corrs):
    # Use tril_indices to access the lower triangle excluding the diagonal
    i, j = np.tril_indices_from(corrs, k=-1)
    return np.mean(corrs[i, j])

def median_pairwise_correlation(corrs):
    i, j = np.tril_indices_from(corrs, k=-1)
    return np.median(corrs[i, j])

def min_pairwise_correlation(corrs):
    i, j = np.tril_indices_from(corrs, k=-1)
    return np.min(corrs[i, j])
    
def save_results_dataframe(output_dir, results):
    """Saves a DataFrame with main results of the analysis."""
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'analysis_results.csv'), index=False)
    print(f"Results saved to {output_dir}/analysis_results.csv")

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

def save_plots(output_dir, all_results, report_file):
    """Saves loss plots for the factor model."""

    print(f"All results object looks like:")
    losses = [result[0]["losses"][-1] for result in all_results]
    num_epochs = [len(result[0]["losses"]) for result in all_results]

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Initialization")
    plt.ylabel("Final Loss")
    plt.scatter(range(len(losses)), losses)
    plt.scatter(np.argmin(losses), np.min(losses), color="red")
    for i in range(len(losses)):
        plt.text(i, losses[i], str(num_epochs[i]), size="small")
    plt.savefig(os.path.join(output_dir, 'losses.png'))
    plt.close()

    log_to_report(report_file, f"Final Losses: {losses}")
    log_to_report(report_file, f"Number of Epochs per Initialization: {num_epochs}")

def calculate_and_plot_correlations(assign_matrices, output_dir, report_file):
    """Calculates and plots the correlation matrix for assignments."""
    corrs, matchings = cost_correlation_assign.compare_assignments(assign_matrices)
    
    plt.figure()
    sns.clustermap(corrs, annot=True)
    plt.savefig(os.path.join(output_dir, 'assignment_correlations.png'), bbox_inches='tight')
    plt.close()

    # Calculate and report correlation metrics
    avg_corr = average_pairwise_correlation(corrs)
    median_corr = median_pairwise_correlation(corrs)
    min_corr = min_pairwise_correlation(corrs)
    
    report = (f"Assignment Correlation Report:\n"
              f"Average Pairwise Correlation: {avg_corr:.4f}\n"
              f"Median Pairwise Correlation: {median_corr:.4f}\n"
              f"Minimum Pairwise Correlation: {min_corr:.4f}\n")
    
    log_to_report(report_file, report)
    return avg_corr, median_corr, min_corr

def plot_umap(latent_space, adata_input, output_dir, plot_name='umap.png', silhouette=None):
    """Plots UMAP with cell types and saves the plot."""
    # Create cell type colors
    cell_types_series = pd.Series(adata_input.obs.cell_type.astype(str).values)
    cell_types_unique = np.unique(cell_types_series)
    cell_type_colors = sns.color_palette("tab20", len(cell_types_unique))
    cell_type_dict = dict(zip(cell_types_unique, cell_type_colors))
    cell_colors = cell_types_series.map(cell_type_dict).values
    
    # Create the UMAP plot
    plt.figure(figsize=(6, 6))
    
    if silhouette is None:
        silhouette = silhouette_score(latent_space, adata_input.obs.cell_type.values, metric='euclidean')
    
    # Run UMAP first on embedding! 
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(latent_space)
    
    # Scatter plot for UMAP
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=cell_colors, alpha=0.5)
    plt.text(0.05, 0.95, f'Silhouette Score: {silhouette:.2f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top')
    
    # Add legend
    legend_handles = [mpatches.Patch(color=color, label=cell_type) for cell_type, color in cell_type_dict.items()]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, plot_name), bbox_inches='tight')
    plt.close()

    return silhouette

def plot_clustermap(embedding, adata_input, output_dir, plot_name='clustermap.png'):
    """Plots clustermap with cell type coloring."""
    cell_types_series = pd.Series(adata_input.obs.cell_type.astype(str).values)
    cell_types_unique = np.unique(cell_types_series)
    cell_type_colors = sns.color_palette("tab20", len(cell_types_unique))
    cell_type_dict = dict(zip(cell_types_unique, cell_type_colors))
    cell_colors = cell_types_series.map(cell_type_dict).values

    sns.clustermap(
        data=embedding,
        annot=False,
        yticklabels=False,
        figsize=(8, 8),
        row_colors=cell_colors,
        cbar_kws={'label': 'Embeddings'}
    )
    plt.title('Clustermap of Embeddings', fontsize=14)
    plt.savefig(os.path.join(output_dir, plot_name), bbox_inches='tight')
    plt.close()

def get_NMF(adata_input, K, output_dir, true_juncs_layer="Junction_Counts", true_clusts_layer="Cluster_Counts"):
    
    """Evaluates model performance with different masking strategies and stores the results."""

    # Step 1: Evaluate MiniBatch NMF as a baseline
    W, H = mask.run_minibatch_nmf_baseline(adata_input, n_components=K, true_juncs_layer=true_juncs_layer,true_clusts_layer=true_clusts_layer)

    # Step 2: Calculate silhouette score using NMF-based embeddings (H) and cell type labels
    cell_types = adata_input.obs['cell_type'].values
    NMF_silhouette = silhouette_score(W, cell_types)  # Silhouette score
    
    # Step 3: Plot clustermap of NMF-based embeddings
    plot_clustermap(W, adata_input, output_dir, plot_name='nmf_clustermap.png')

    # Step 4: UMAP projection for dimensionality reduction
    plot_umap(W, adata_input, output_dir, plot_name='nmf_umap.png', silhouette=NMF_silhouette)

    # Return the silhouette score for evaluation
    return NMF_silhouette

def compute_and_plot_albf(adata_input, psis_mus, psis_loc, psis, pi, output_dir, K, report_file):
    
    # Compute ALBF
    albf, l0 = differential_splicing.compute_albf(psis_mus, psis_loc + 1e-9, torch.tensor(pi))
    l0 = l0.detach().cpu()
    albf = albf.detach().cpu()
    
    # Prepare dataframes
    albf_df = pd.DataFrame(albf, columns=["ALBF"])
    albf_df["junction_id_index"] = range(albf_df.shape[0])
    
    psis_df = pd.DataFrame(psis.T)
    psis_df["junction_id_index"] = psis_df.index
    psis_df = psis_df.merge(albf_df, on=["junction_id_index"])    
    return psis_df

def plot_pi_barplot(pi, output_dir):
    pi_df = pd.DataFrame(pi, columns=["pi"])
    pi_df["Factor"] = "Factor" + pi_df.index.astype(str)
    pi_df = pi_df.sort_values(by="pi", ascending=False)

    plt.figure(figsize=(6, 6))
    sns.barplot(x="Factor", y="pi", data=pi_df, palette="viridis")
    plt.xlabel("Factor")
    plt.xticks(rotation=90, size=8)
    plt.ylabel("pi")
    plt.title("Overall contribution of each factor to cell population")
    plt.savefig(os.path.join(output_dir, 'pi_barplot.png'))
    plt.close()

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

def prepare_output_directory(K_use, use_global_prior, input_conc, num_inits, num_epochs, cell_type_column):
    """Prepares the output directory with timestamp and relevant parameters."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cell_type_str = f"CellType_{cell_type_column}" if cell_type_column else "NoCellType"
    use_global_prior_str = "GlobalPrior" if use_global_prior else "NoGlobalPrior"
    
    # Handle input_conc for directory naming
    input_conc_prior_str = "LearnedConc" if input_conc is None else f"ConcPrior_{input_conc}"
    
    output_dir = (f"./analysis_{timestamp}_K_{K_use}_"
                  f"Prior_{use_global_prior_str}_NumEpochs_{num_epochs}_"
                  f"{input_conc_prior_str}_Inits_{num_inits}_"
                  f"{cell_type_str}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_parameters(output_dir, args, K, input_conc):
    """Saves run parameters as a JSON file."""
    parameters = {
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

    # Prepare the output directory with timestamp
    output_dir = prepare_output_directory(K_use, use_global_prior, input_conc, num_inits, num_epochs, cell_type_column)
    report_file = create_report_file(output_dir)

    # Load data and set K
    log_to_report(report_file, f"Loading anndata file from {input_path}")
    adata = load_adata(input_path)
    adata.var["junction_id_index"] = adata.var.index.astype(int)
    adata.var.reset_index(drop="junction_id_index", inplace=True)

    K = K_use

    # Save run parameters as JSON
    save_parameters(output_dir, args, K, input_conc)

    # Preprocess the data
    adata_filtered = adata.copy()

    # Check if layers are in correct format
    if isinstance(adata_filtered.layers.get("Junction_Counts"), csr_matrix):
        adata_filtered.layers["Junction_Counts"] = adata_filtered.layers["Junction_Counts"].tocoo()

    if isinstance(adata_filtered.layers.get("Cluster_Counts"), csr_matrix):
        adata_filtered.layers["Cluster_Counts"] = adata_filtered.layers["Cluster_Counts"].tocoo()

    # Convert back to sparse tensors for model input
    cell_index_tensor, junc_index_tensor, my_data = llc.make_torch_adata(
        adata_filtered, 
        cluster_layer="Cluster_Counts", 
        junction_layer="Junction_Counts", 
        **float_type
    )

    # Run the factor model
    log_to_report(report_file, f"Running factor model with K = {K}!")
    all_results, all_params = run_factor_model(my_data, K, device, use_global_prior, input_conc, num_inits, num_epochs, lr)
    save_plots(output_dir, all_results, report_file)

    # Calculate and plot correlations of assignment matrices
    log_to_report(report_file, "Calculating correlations between initializations.")
    assign_matrices = [result[0]["summary_stats"]["assign"]["mean"] for result in all_results]
    avg_corr, median_corr, min_corr = calculate_and_plot_correlations(assign_matrices, output_dir, report_file)

    # Select the best initialization based on the loss
    best_init = np.argmin([result[0]["losses"][-1] for result in all_results])
    latent_vars = all_results[best_init][0]['summary_stats']
    
    # Differential Splicing Analysis
    log_to_report(report_file, "Running differential splicing analysis.")
    psis_mus = all_params[best_init]["AutoGuideList.0.loc"].reshape(K, latent_vars["psi"]["mean"].shape[1])
    psis_loc = all_params[best_init]["AutoGuideList.0.scale"].reshape(K, latent_vars["psi"]["mean"].shape[1])
    pi = latent_vars["pi"]["mean"]
    assign_post = latent_vars["assign"]["mean"]

    # If input_conc was None extract it from latent variables 
    if input_conc == None:
        input_conc = latent_vars["bb_conc"]["mean"]
    else:
        input_conc = "infinity"

    # Get metrics from ALBF
    psis_df = compute_and_plot_albf(adata, psis_mus, psis_loc, latent_vars["psi"]["mean"], pi, output_dir, K, report_file)
    
    # Add psis_df to Anndata object 
    psis_df.rename(columns={i: f"cell_state_{i}" for i in range(K)}, inplace=True)
    adata.var = pd.concat([adata.var, psis_df.set_index('junction_id_index')], axis=1)

    # Calculate silhouette score
    silhouette_avg = plot_umap(assign_post, adata, output_dir)
    plot_pi_barplot(pi, output_dir)
    plot_clustermap(assign_post, adata, output_dir)

    # Get silhouette score from NMF 
    silhouette_NMF = get_NMF(adata, K=K, output_dir=output_dir)

    # Step 1: Define the columns for the DataFrame
    columns = [
        "K", "use_global_prior", "input_conc", "learning_rate", 
        "avg_corr", "median_corr", "min_corr", 
        "silhouette_avg", "cell_type_column", "silhouette_NMF", "input_conc"
    ]

    if cell_type_column == None:
        cell_type_column = "None"

    # Step 2: Create a DataFrame with a single row
    results_df = pd.DataFrame([[
        K, use_global_prior, input_conc, lr, 
        avg_corr, median_corr, min_corr, 
        silhouette_avg, cell_type_column, silhouette_NMF, input_conc
    ]], columns=columns)

    # Step 3: Save the DataFrame to a CSV file in the output directory
    results_path = os.path.join(output_dir, "final_results.csv")
    results_df.to_csv(results_path, index=False)

    # Save the `adata_input` AnnData object to the output directory
    adata_output_path = os.path.join(output_dir, "adata_input.h5ad")
    adata.var.columns = adata.var.columns.astype(str)

    # Check and convert both 'Cluster_Counts' and 'Junction_Counts' layers from COO to CSR (or CSC)
    for layer in ['Cluster_Counts', 'Junction_Counts']:
        if isinstance(adata.layers[layer], scipy.sparse.coo_matrix):
            adata.layers[layer] = adata.layers[layer].tocsr()  # Convert to CSR or use .tocsc() if you prefer

    # Save the AnnData object for this particular analysis 
    adata.write(adata_output_path)

    log_to_report(report_file, f"AnnData saved to {adata_output_path}")

    # Save all latent variables (i.e., trained model parameters) to a pickle file
    latent_vars_output_path = os.path.join(output_dir, "latent_vars.pkl")
    with open(latent_vars_output_path, "wb") as f:
        pickle.dump(latent_vars, f)
    log_to_report(report_file, f"Latent variables saved to {latent_vars_output_path}")


    print(f"Results saved to {results_path}")

    report_file.close()

if __name__ == "__main__":
    main()