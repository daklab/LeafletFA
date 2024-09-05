import argparse

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run differential splicing analysis.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input .h5ad file.')
    parser.add_argument('--proportion_negative', type=float, default=0.5, help='Proportion of negatively spliced junctions.')
    parser.add_argument('--cell_type_column', type=str, default=None, help='Column name for cell types in the AnnData object.')
    parser.add_argument('--K_use', type=int, default=3, help='Number of clusters to use if cell_type_column is None.')
    parser.add_argument('--use_global_prior', action='store_true', help='Whether to use global prior in the factor model.')
    parser.add_argument('--input_conc_prior', type=str, default="inf", help='Input concentration prior value for the factor model.')
    parser.add_argument('--num_inits', type=int, default=3, help='Number of times to randomly initialize our factor model.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to run model training.')
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

sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/evaluations')
import cost_correlation_assign
import differential_splicing

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

float_type = {"device": device, "dtype": torch.float}
if device == torch.device('cuda'):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Create the report file early in the script
def create_report_file(output_dir):
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    return open(report_path, 'w')

def log_to_report(report_file, text):
    report_file.write(text + '\n')
    print(text)

# Load input data
def load_adata(input_path):
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

def preprocess_adata(adata, cell_type_column):
    '''
    
    '''
    if cell_type_column is not None:
        
        print(f"Filtering ATSEs to remove those very unevenly distributed across cell types!")
        cluster_counts = adata.layers["Cluster_Counts"]
        cell_types = adata.obs[cell_type_column].values
        unique_cell_types = np.unique(cell_types)
        unique_clusters = adata.var_names

        expression_counts = pd.DataFrame(0, index=unique_cell_types, columns=unique_clusters)
        for cell_type in tqdm(unique_cell_types):
            cells_in_type = (cell_types == cell_type)
            counts_in_type = cluster_counts.toarray()[cells_in_type, :].sum(axis=0)
            expression_counts.loc[cell_type] = counts_in_type

        non_zero_counts = (expression_counts >= 5).sum(axis=0)
        threshold = len(expression_counts) * 0.2
        filtered_clusters = non_zero_counts[non_zero_counts > threshold].index

        filtered_expression_counts = expression_counts[filtered_clusters]
        juncs_keep = list(filtered_expression_counts)
        adata_filtered = adata[:, juncs_keep].copy()
        adata_filtered.var['junction_id_index'] = np.arange(adata_filtered.n_vars)
        adata_filtered.var_names = adata_filtered.var['junction_id_index'].astype(str)
        return adata_filtered
    else:
        return adata.copy()
    
def simulate_and_prepare(adata, K, float_type, proportion_negative, cell_type_column):
    return sim.simulate_and_prepare_data(adata, K, float_type, proportion_negative, cell_type_column)

def run_factor_model(my_data, K, float_type, device, use_global_prior, input_conc, num_inits, num_epochs):

    # Convert sparse matrices to COO format, ensure they are on the GPU, and follow the specified float type
    full_y_tensor = my_data.ycount_lookup.to_sparse_coo()
    full_total_counts_tensor = my_data.tcount_lookup.to_sparse_coo()
    print(f"Running on device: {device}")

    # Ensure that input_conc is also on the correct device and follows the specified float type

    print(f"use_global_prior: {use_global_prior}") 
    
    # Run the factor model
    all_results, variable_sizes = factor_model.main(
        full_y_tensor, 
        full_total_counts_tensor, 
        num_initializations=num_inits, 
        use_global_prior=use_global_prior, 
        K=K, 
        lr=0.1, 
        input_conc_prior=input_conc, 
        loss_plot=False, 
        num_epochs=num_epochs, 
        save_to_file=False
    )

    return all_results

def save_plots(output_dir, all_results, report_file):
    losses = [result["losses"][-1] for result in all_results]
    num_epochs = [len(result["losses"]) for result in all_results]

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
    
    with open(os.path.join(output_dir, 'correlation_report.txt'), 'w') as f:
        f.write(report)
    
    log_to_report(report_file, report)

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
    
    psis_df["learned_difference"] = psis_df.iloc[:, :K].std(axis=1)

    juncs_clusts_labs = adata_input.var[["junction_id_index", "difference", "true_label", "Cluster"]]
    psis_df = psis_df.merge(juncs_clusts_labs)
    
    # Plot ALBF score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=psis_df, x='ALBF', hue='true_label', kde=True, bins=30, alpha=0.5)
    plt.title('Distribution of ALBF Scores by True Label')
    plt.xlabel('ALBF Score')
    plt.ylabel('Density')
    plt.savefig(os.path.join(output_dir, 'ALBF_score_distribution.png'))
    plt.close()

    # Plot ALBF vs learned_difference
    plt.figure(figsize=(5, 5))
    sns.scatterplot(data=psis_df, x="learned_difference", y="ALBF")
    correlation_learned_diff = psis_df["ALBF"].corr(psis_df["learned_difference"], method="spearman")
    plt.title(f'Spearman correlation: {correlation_learned_diff:.2f}')
    plt.savefig(os.path.join(output_dir, 'albf_vs_learned_difference.png'))
    plt.close()
    print(f"Spearman correlation between ALBF and learned_difference: {correlation_learned_diff}")

    # Plot ALBF vs difference
    plt.figure(figsize=(5, 5))
    sns.scatterplot(data=psis_df, x="difference", y="ALBF")
    correlation_diff = psis_df["ALBF"].corr(psis_df["difference"], method="spearman")
    plt.title(f'Spearman correlation: {correlation_diff:.2f}')
    plt.savefig(os.path.join(output_dir, 'albf_vs_difference.png'))
    plt.close()
    print(f"Spearman correlation between ALBF and difference: {correlation_diff}")

    # Prepare the report
    report = (
        f"ALBF Score Evaluation:\n"
        f"Spearman correlation between ALBF and learned_difference: {correlation_learned_diff:.4f}\n"
        f"Spearman correlation between ALBF and difference: {correlation_diff:.4f}\n"
    )

    with open(os.path.join(output_dir, 'ALBF_report.txt'), 'w') as f:
        f.write(report)

    log_to_report(report_file, report)    

def plot_umap(assign_post, adata_input, output_dir):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(assign_post)

    cell_types_series = pd.Series(adata_input.obs.cell_type.astype(str).values)
    cell_types_unique = np.unique(cell_types_series)
    cell_type_colors = sns.color_palette("tab20", len(cell_types_unique))
    cell_type_dict = dict(zip(cell_types_unique, cell_type_colors))
    cell_colors = cell_types_series.map(cell_type_dict).values

    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.figure(figsize=(6, 6))

    silhouette_avg = silhouette_score(assign_post, adata_input.obs.cell_type.values, metric='euclidean')

    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=cell_colors, alpha=0.5)
    plt.text(0.05, 0.95, f'Silhouette Score: {silhouette_avg:.2f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top')

    legend_handles = [mpatches.Patch(color=color, label=cell_type) for cell_type, color in cell_type_dict.items()]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(output_dir, 'umap_plot.png'), bbox_inches='tight')
    plt.close()

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

def plot_clustermap(assign_post, adata_input, output_dir):
    cell_types_series = pd.Series(adata_input.obs.cell_type.astype(str).values)
    cell_types_unique = np.unique(cell_types_series)
    cell_type_colors = sns.color_palette("tab20", len(cell_types_unique))
    cell_type_dict = dict(zip(cell_types_unique, cell_type_colors))
    cell_colors = cell_types_series.map(cell_type_dict).values

    sns.clustermap(
        data=assign_post,
        annot=False,
        yticklabels=False,
        figsize=(8, 8),
        row_colors=cell_colors,
        cbar_kws={'label': 'Post assignment'}
    )
    plt.savefig(os.path.join(output_dir, 'clustermap.png'), bbox_inches='tight')
    plt.close()

def main():
    
    args = parse_arguments()

    # Handle different types for input_conc_prior
    if isinstance(args.input_conc_prior, str):
        if args.input_conc_prior.lower() == "inf":
            input_conc = torch.tensor(np.inf, **float_type)
        else:
            input_conc = torch.tensor(float(args.input_conc_prior), **float_type)
    elif isinstance(args.input_conc_prior, float):
        if args.input_conc_prior == float('inf'):
            input_conc = torch.tensor(np.inf, **float_type)
        else:
            input_conc = torch.tensor(args.input_conc_prior, **float_type)
    elif isinstance(args.input_conc_prior, torch.Tensor):
        # Assume the tensor is already correctly typed and just ensure it’s on the correct device
        input_conc = args.input_conc_prior.to(**float_type)
    else:
        raise ValueError("Unsupported type for input_conc_prior")

    # Convert the string "None" to the actual None value
    if args.cell_type_column == "None":
        args.cell_type_column = None

    # Extract and summarize parameters
    proportion_negative = args.proportion_negative
    cell_type_column = args.cell_type_column 
    K_use = args.K_use
    input_path = args.input_path
    use_global_prior = args.use_global_prior  # This remains a boolean
    use_global_prior_str = "GlobalPrior" if use_global_prior else "NoGlobalPrior"
    input_conc_prior = input_conc
    input_conc_prior_str = f"ConcPrior_{input_conc_prior}"
    num_inits = args.num_inits
    num_epochs = args.num_epochs 

    # Add cell_type_column to output directory name, if provided
    cell_type_str = f"CellType_{cell_type_column}" if cell_type_column else "NoCellType"

    # Timestamp for analysis folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = (f"./analysis_{timestamp}_PropNeg_{proportion_negative}_K_{K_use}_"
              f"Prior_{use_global_prior_str}_NumEpochs_{num_epochs}_"
              f"Conc_{input_conc_prior_str}_Inits_{num_inits}_"
              f"{cell_type_str}")
    
    os.makedirs(output_dir, exist_ok=True)

    # Create the report file
    report_file = create_report_file(output_dir)

    # Load data
    log_to_report(report_file, f"Loading anndata file from {input_path}")
    adata = load_adata(input_path)
    
    K = adata.obs[cell_type_column].nunique() if cell_type_column else K_use
    print(f"K is set to: {K} for the analysis...")

    # Extract and summarize parameters
    
    parameters = {
        "proportion_negative": args.proportion_negative,
        "cell_type_column": args.cell_type_column,
        "K_use": K,
        "input_path": args.input_path,
        "use_global_prior": args.use_global_prior,
        "input_conc_prior": input_conc.item() if torch.is_tensor(input_conc) else input_conc,  # Convert tensor to float
        "num_inits": args.num_inits,
        "num_epochs": args.num_epochs,
        "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    # Save parameters dictionary as a JSON file
    json_path = os.path.join(output_dir, 'parameters.json')
    with open(json_path, 'w') as json_file:
        json.dump(parameters, json_file, indent=4)

    print(f"Parameters saved to {json_path}")

    adata_filtered = preprocess_adata(adata, cell_type_column)

    # Simulate data 
    log_to_report(report_file, f"Simulating splice junction counts with proportions of events set to be negative: {proportion_negative}!")
    my_data, adata_input = simulate_and_prepare(adata_filtered, K, float_type, proportion_negative, cell_type_column)

    # Run the factor model
    log_to_report(report_file, f"Running Leaflet factor model on simulated data where K = {K}!")
    print(f"Running Leaflet factor model on simulated data where K = {K}!")
    print(f"Input conc prior is {input_conc_prior}")
    print(f"Global prior is set to: {use_global_prior}")
    
    #input_conc_prior = torch.tensor(np.inf, **float_type)

    all_results = run_factor_model(my_data, K, float_type, device, use_global_prior, input_conc_prior, num_inits=num_inits, num_epochs=num_epochs)
    save_plots(output_dir, all_results, report_file)

    # Calculate correlations of assignment matrices
    log_to_report(report_file, "Calculate correlation between the different initializations!")
    assign_matrices = [result["summary_stats"]["assign"]["mean"] for result in all_results]
    calculate_and_plot_correlations(assign_matrices, output_dir, report_file)

    # Differential Splicing Analysis
    print(f"Extracting latent variables using the best initialization!")
    best_init = np.argmin([result["losses"][-1] for result in all_results])
    latent_vars = all_results[best_init]['summary_stats']
    
    J = latent_vars["psi"]["mean"].shape[1]

    # Extract latent variables
    psis_mus = pyro.get_param_store()["AutoGuideList.0.loc"].reshape(K, J)
    psis_loc = pyro.get_param_store()["AutoGuideList.0.scale"].reshape(K, J)
    pi = latent_vars["pi"]["mean"]
    dir_conc = latent_vars["dir_conc"]["mean"]
    assign_post = latent_vars["assign"]["mean"]
    psis = latent_vars["psi"]["mean"]
    a = latent_vars["a"]["mean"] 
    b = latent_vars["b"]["mean"]

    print("Inferred Parameters:")
    print(f"The inferred concentration parameter is: {dir_conc}")
    print(f"The inferred pi parameter is: {pi}")
    print(f"a: {a}")
    print(f"b: {b}")

    print(f"Running differential splicing analysis for each splice junction using ALBF!")
    log_to_report(report_file, "Running differential splicing analysis for each splice junction using ALBF!")

    compute_and_plot_albf(adata_input, psis_mus, psis_loc, psis, pi, output_dir, K, report_file)
    plot_umap(assign_post, adata_input, output_dir)
    plot_pi_barplot(pi, output_dir)
    plot_clustermap(assign_post, adata_input, output_dir)
    # Close the report file
    report_file.close()

if __name__ == "__main__":
    main()