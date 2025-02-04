import argparse
import pickle
import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import anndata as ad
from importlib import reload
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import pyro 
import umap.umap_ as umap
from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import scipy.sparse as sp 

# Import the factor model and simulation source code
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/beta-dirichlet-factor')
import factor_model
reload(factor_model)

import differential_splicing
reload(differential_splicing)

# Simulation source code
sys.path.append("/gpfs/commons/home/kisaev/Leaflet-private/src/simulation/")
import simulate_counts as sim 
reload(sim)

# Evaluation source code
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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run differential splicing analysis.')

    # Required arguments
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input .h5ad file.')
    parser.add_argument('--ATSE_file', type=str, help='File containing ATSE annotations from the previous Leaflet processing step.')
    parser.add_argument('--save_anndata', action='store_true', help='Indicate whether to save simulted anndata file.')

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
    parser.add_argument('--mask_perc', type=float, default=0.1, help='Percentage of data (nonzero cluster counts) to mask.')
    parser.add_argument('--proportion_negative', type=float, default=0.5, help='Proportion of negatively spliced junctions.')

    return parser.parse_args()

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

def preprocess_adata(adata, cell_type_column, cluster_layer="Cluster_Counts"):
    """
    Preprocesses the AnnData object by filtering unevenly distributed splicing events (ATSEs).
    
    If a cell type column is specified, the function filters out junctions that have
    low expression across the cell types.
    """
    if cell_type_column is not None:
        
        print(f"Filtering ATSEs to remove those very unevenly distributed across cell types!")
        cluster_counts = adata.layers[cluster_layer]
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
    """Simulates and prepares the data for factor model input."""
    return sim.simulate_and_prepare_data(adata, K, float_type, proportion_negative, cell_type_column, gen_model_input=True)

def save_results_dataframe(output_dir, results):
    """Saves a DataFrame with main results of the analysis."""
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'analysis_results.csv'), index=False)
    print(f"Results saved to {output_dir}/analysis_results.csv")

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
    
    # Set font size for axis labels and tick marks
    plt.xlabel('UMAP 1', fontsize=14)  # Increase the font size for x-axis label
    plt.ylabel('UMAP 2', fontsize=14)  # Increase the font size for y-axis label
    plt.tick_params(axis='both', which='major', labelsize=12)  # Increase the font size for tick marks

    # Add legend
    legend_handles = [mpatches.Patch(color=color, label=cell_type) for cell_type, color in cell_type_dict.items()]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot
    # plt.savefig(os.path.join(output_dir, plot_name), bbox_inches='tight')
    # Save the plot as a PDF
    pdf_filename = os.path.join(output_dir, f"{plot_name}.pdf")
    plt.savefig(pdf_filename, bbox_inches='tight', format='pdf')
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

def get_NMF(adata_input, K, output_dir, true_juncs_layer="cell_by_junction_matrix", true_clusts_layer="cell_by_cluster_matrix"):
    
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
    l0 = []
    albf_values = []

    # Reshape psis_mus and psis_loc to be (J, K) instead of (K, J)
    psis_mus = psis_mus.T
    psis_loc = psis_loc.T
    J = psis_mus.shape[0]

    for j in range(J):  # Iterate over all J junctions
        albf, log_pj_h0 = differential_splicing.compute_albf(psis_mus[j], psis_loc[j], pi)
        l0.append(log_pj_h0.item())
        albf_values.append(albf.item())

    l0 = torch.tensor(l0).detach().cpu()
    albf_values = torch.tensor(albf_values).detach().cpu()

    # Prepare dataframes
    # albf_df = pd.DataFrame(albf.flatten(), columns=["ALBF"])
    albf_df = pd.DataFrame(albf_values.numpy().flatten(), columns=["ALBF"])

    albf_df["junction_id_index"] = range(albf_df.shape[0])
    print(albf_df.head())
    
    psis_df = pd.DataFrame(psis.T)
    psis_df["junction_id_index"] = psis_df.index
    psis_df = psis_df.merge(albf_df, on=["junction_id_index"])
    
    juncs_clusts_labs = adata_input.var[["junction_id_index", "difference", "true_label", "Cluster"]]
    psis_df = psis_df.merge(juncs_clusts_labs)
    
    # Convert "true_label" to binary format: 1 for "positive", 0 for "negative"
    psis_df['true_label_binary'] = psis_df['true_label'].apply(lambda x: 1 if x == 'positive' else 0)
    true_labels = psis_df['true_label_binary']
    albf_scores = psis_df['ALBF']
    psis_df["delta_est"] = np.abs(psis_df[1] - psis_df[0])

    # Step 1: Calculate ROC-AUC score
    auc_score = roc_auc_score(true_labels, albf_scores)
    print(f"ROC-AUC Score: {auc_score:.4f}")

    # Step 2: Plot ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, albf_scores)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # Step 3: Set threshold for ALBF classification (e.g., You can use the threshold where TPR ≈ 0.5 for balanced classification)
    optimal_idx = np.argmax(tpr - fpr)  # Maximize TPR - FPR to find the best threshold
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold for ALBF classification: {optimal_threshold}")

    # Step 4: Classify based on optimal threshold and calculate confusion matrix
    psis_df['predicted_label'] = psis_df['ALBF'].apply(lambda x: 1 if x >= optimal_threshold else 0)
    
    cm = confusion_matrix(true_labels, psis_df['predicted_label'])
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    accuracy = accuracy_score(true_labels, psis_df['predicted_label'])
    precision = precision_score(true_labels, psis_df['predicted_label'])
    recall = recall_score(true_labels, psis_df['predicted_label'])

    # Step 5: Plot ALBF score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=psis_df, x='ALBF', hue='true_label', kde=True, bins=30, alpha=0.5)
    plt.title('Distribution of ALBF Scores by True Label')
    plt.xlabel('ALBF Score')
    plt.ylabel('Density')
    plt.savefig(os.path.join(output_dir, 'ALBF_score_distribution.png'))
    plt.close()

    # Step 5: Compute and plot the Precision-Recall curve
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(true_labels, albf_scores)
    average_precision = average_precision_score(true_labels, albf_scores)

    plt.figure(figsize=(5, 5))
    plt.plot(recall_vals, precision_vals, label=f'Precision-Recall curve (AP = {average_precision:.2f})') 
    # Set font size for axis labels, title, and legend
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    # Increase font size for x-axis and y-axis tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    # Save the plot as both PNG and PDF
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.pdf'), bbox_inches='tight', format='pdf')
    plt.close()

    # Step 6: Plot ALBF vs difference
    plt.figure(figsize=(5, 5))
    sns.scatterplot(data=psis_df, x="difference", y="ALBF")
    correlation_diff_albf = psis_df["ALBF"].corr(psis_df["difference"], method="spearman")
    plt.title(f'Spearman correlation: {correlation_diff_albf:.2f}')
    plt.savefig(os.path.join(output_dir, 'albf_vs_difference.png'))
    plt.close()

    plt.figure(figsize=(5, 5))
    sns.scatterplot(data=psis_df, x="difference", y="delta_est")
    correlation_diff_delta = psis_df["delta_est"].corr(psis_df["difference"], method="spearman")
    plt.title(f'Spearman correlation: {correlation_diff_delta:.2f}')
    plt.xlabel("Simualte Delta PSI", fontsize=14)
    plt.ylabel("Estimated Delta PSI", fontsize=14)
    # Increase font size for x-axis and y-axis tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(os.path.join(output_dir, 'est_deltapsi_vs_difference.png'))
    plt.savefig(os.path.join(output_dir, 'est_deltapsi_vs_difference.pdf'), bbox_inches='tight', format='pdf')
    plt.close()

    # Step 7: Print and report the classification performance
    print(f"Spearman correlation between ALBF and difference: {correlation_diff_albf}")
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print(f"Precision-Recall AUC (Average Precision): {average_precision:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Confusion Matrix: \n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Misidentified as positive (False Positives): {fp}")
    print(f"Misidentified as negative (False Negatives): {fn}")
    
    # Prepare the report
    report = (
        f"ALBF Score Evaluation:\n"
        f"Spearman correlation between ALBF and difference: {correlation_diff_albf:.4f}\n"
        f"ROC-AUC Score: {auc_score:.4f}\n"
        f"Precision-Recall AUC (Average Precision): {average_precision:.4f}\n"
        f"Optimal Threshold: {optimal_threshold:.4f}\n"
        f"Confusion Matrix: \n{cm}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall (Sensitivity): {recall:.4f}\n"
        f"Misidentified as positive (False Positives): {fp}\n"
        f"Misidentified as negative (False Negatives): {fn}\n"
    )

    with open(os.path.join(output_dir, 'ALBF_report.txt'), 'w') as f:
        f.write(report)

    # Write psis_df to a CSV file
    psis_df.to_csv(os.path.join(output_dir, 'ALBF_scores.csv'), index=False) 

    return correlation_diff_albf, correlation_diff_delta, auc_score, optimal_threshold, accuracy, precision, recall, fp, fn

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

def prepare_output_directory(proportion_negative, K_use, use_global_prior, input_conc, num_inits, num_epochs, cell_type_column, waypoint):
    """Prepares the output directory with timestamp and relevant parameters."""

    # Generate a random number to ensure uniqueness (e.g., 5-digit random integer)
    random_number = np.random.randint(10, 100)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cell_type_str = f"CellType_{cell_type_column}" if cell_type_column else "NoCellType"
    use_global_prior_str = "GlobalPrior" if use_global_prior else "NoGlobalPrior"
    waypoint_str = "UsingWaypoints" if waypoint else "NoWaypoints"
    
    # Handle input_conc for directory naming
    input_conc_prior_str = "LearnedConc" if input_conc is None else f"ConcPrior_{input_conc}"
    
    # Construct the output directory name with random number appended
    output_dir = (f"./analysis_{timestamp}_PropNeg_{proportion_negative}_K_{K_use}_{waypoint_str}_"
                  f"Prior_{use_global_prior_str}_NumEpochs_{num_epochs}_"
                  f"{input_conc_prior_str}_Inits_{num_inits}_"
                  f"{cell_type_str}_Random_{random_number}")
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_parameters(output_dir, args, K, input_conc):
    """Saves run parameters as a JSON file."""
    parameters = {
        "proportion_negative": args.proportion_negative,
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
    proportion_negative = args.proportion_negative
    K_use = args.K_use
    input_path = args.input_path
    use_global_prior = args.use_global_prior
    num_inits = args.num_inits
    num_epochs = args.num_epochs
    lr = args.lr  # Learning rate
    ATSE_file = args.ATSE_file 
    waypoints_use = args.waypoints_use
    brain_only = args.brain_only
    save_anndata = args.save_anndata
    
    # Read in the intron cluster file 
    print("Reading in obtained intron cluster (ATSE file!)")
    intron_clusts = pd.read_csv(ATSE_file, sep="}")
    genes = intron_clusts[["gene_id", "gene_name"]].drop_duplicates()

    if brain_only:
        print(f"Running model only on brain cells!")
    else:
        print(f"Running on all tissues!")

    # Prepare the output directory with timestamp
    output_dir = prepare_output_directory(proportion_negative, K_use, use_global_prior, input_conc, num_inits, num_epochs, cell_type_column, waypoints_use)
    report_file = create_report_file(output_dir)

    # Load data 
    log_to_report(report_file, f"Loading anndata file from {input_path}")

    adata = load_adata(input_path)
    adata.var = pd.merge(adata.var, genes[['gene_id', 'gene_name']], how='left', on='gene_id')

    # Set K to number of unique cell types if column specified otherwise user the provided K
    K = adata.obs[cell_type_column].nunique() if cell_type_column else K_use

    # Save run parameters as JSON
    save_parameters(output_dir, args, K, input_conc)

    # Preprocess the data
    adata_filtered = preprocess_adata(adata, cell_type_column, "cell_by_cluster_matrix")

    # Simulate data
    log_to_report(report_file, f"Simulating splice junction counts with {proportion_negative} proportion negative!")
    full_y_tensor, full_total_counts_tensor, adata_input, cell_type_psi_df = simulate_and_prepare(adata_filtered, K, float_type, proportion_negative, cell_type_column)

    # Write cell_type_psi_df to a CSV file 
    cell_type_psi_df_path = os.path.join(output_dir, 'cell_type_psi_df.csv')
    cell_type_psi_df.to_csv(cell_type_psi_df_path, index=False)

    if save_anndata:
        # Save the `adata_input` AnnData object to the output directory
        adata_output_path = os.path.join(output_dir, "adata_input.h5ad")
        adata_input.var.columns = adata_input.var.columns.astype(str)

        for layer_name, layer_data in adata_input.layers.items():
            if isinstance(layer_data, sp.coo_matrix):  
                print(f"Converting layer {layer_name} from COO to CSR format.")
                adata_input.layers[layer_name] = layer_data.tocsr()  # Convert COO to CSR

        # Convert 'junc_ratio' from numpy.matrix to numpy.ndarray if it exists in layers
        if isinstance(adata_input.layers["junc_ratio"], np.matrix):
            adata_input.layers["junc_ratio"] = np.asarray(adata_input.layers["junc_ratio"])

        # Save the AnnData object for this particular analysis 
        adata_input.write(adata_output_path)
        log_to_report(report_file, f"AnnData saved to {adata_output_path}")
        
         # Save tensors to the output directory
        y_tensor_path = os.path.join(output_dir, 'full_y_tensor.pt')
        total_counts_tensor_path = os.path.join(output_dir, 'full_total_counts_tensor.pt')

        torch.save(full_y_tensor, y_tensor_path)
        torch.save(full_total_counts_tensor, total_counts_tensor_path)
    
    log_to_report(report_file, f"Running factor model with K = {K}!")
    all_results, all_params = run_factor_model(adata_input, full_y_tensor, full_total_counts_tensor, K, device, use_global_prior, input_conc, waypoints_use, output_dir, num_inits, num_epochs, lr)
    save_plots(output_dir, all_results, report_file)

    # Calculate and plot correlations of assignment matrices
    log_to_report(report_file, "Calculating correlations between initializations.")
    assign_matrices = [result[0]["summary_stats"]["assign"]["mean"] for result in all_results]
    avg_corr, median_corr, min_corr = calculate_and_plot_correlations(assign_matrices, output_dir, report_file)

    # Select the best initialization based on the loss
    best_init = np.argmin([result[0]["losses"][-1] for result in all_results])
    latent_vars = all_results[best_init][0]['summary_stats']
    best_elbo = all_results[best_init][0]["losses"][-1]

    # Differential Splicing Analysis
    log_to_report(report_file, "Running differential splicing analysis.")
    psis_mus = all_params[best_init]["AutoGuideList.0.loc"].reshape(K, latent_vars["psi"]["mean"].shape[1]) # K by J
    psis_loc = all_params[best_init]["AutoGuideList.0.scale"].reshape(K, latent_vars["psi"]["mean"].shape[1]) # K by J
    pi = latent_vars["pi"]["mean"] # shape (K,)
    assign_post = latent_vars["assign"]["mean"] # shape (N, K)

    # If input_conc was None extract it from latent variables 
    if input_conc == None:
        input_conc = latent_vars["bb_conc"]["mean"]
    else:
        input_conc = "infinity"

    # Get metrics from ALBF
    correlation_diff_albf, correlation_diff_delta, auc_score, optimal_threshold, accuracy, precision, recall, fp, fn = compute_and_plot_albf(adata_input, psis_mus, psis_loc, latent_vars["psi"]["mean"], pi, output_dir, K, report_file)
    
    # Calculate silhouette score
    silhouette_avg = plot_umap(assign_post, adata_input, output_dir)
    plot_pi_barplot(pi, output_dir)
    plot_clustermap(assign_post, adata_input, output_dir)
    
    # Get silhouette score from NMF 
    silhouette_NMF = get_NMF(adata_input, K=K, output_dir=output_dir)
    
    # Step 1: Define the columns for the DataFrame
    columns = [
        "proportion_negative", "K", "use_global_prior", "input_conc", "learning_rate", 
        "avg_corr", "median_corr", "min_corr", "correlation_diff_albf", "correlation_diff_delta", 
        "auc_score", "optimal_threshold", "accuracy", "precision", "recall", 
        "false_positives", "false_negatives", "silhouette_avg", "cell_type_column", 
        "silhouette_NMF", "input_conc"
    ]

    if cell_type_column == None:
        cell_type_column = "None"

    # Step 2: Create a DataFrame with a single row
    results_df = pd.DataFrame([[
            proportion_negative, K, use_global_prior, input_conc, lr, 
            avg_corr, median_corr, min_corr, correlation_diff_albf, correlation_diff_delta, auc_score, 
            optimal_threshold, accuracy, precision, recall, 
            fp, fn, silhouette_avg, cell_type_column, silhouette_NMF, input_conc
        ]], columns=columns)

    # Step 3: Save the DataFrame to a CSV file in the output directory
    results_path = os.path.join(output_dir, "final_results.csv")
    results_df.to_csv(results_path, index=False)

    print(f"Results saved to {results_path}")
    report_file.close()

if __name__ == "__main__":
    main()