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
# Import custom modules
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/beta-dirichlet-factor')
import factor_model
reload(factor_model)

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

def run_factor_model(adata_input, full_y_tensor, full_total_counts_tensor, K, device, use_global_prior, input_conc, waypoints_use, output_dir, num_inits=3, num_epochs=100, lr=0.1):
    """Runs the factor model with specified parameters, including the learning rate."""
    
    print(f"Running on device: {device}")
    print(f"use_global_prior: {use_global_prior}") 
    
    all_results = []
    all_params = []  # Store parameters for each initialization       

    if waypoints_use:
        print(f"Initializing variational parameters with pre-defined PSI and PHI matrices!")
        
        # Dynamically select the correct PSI and PHI based on the value of K
        psi_key = f"psi_init_{K}_waypoints"
        phi_key = f"phi_init_{K}_waypoints"

        if psi_key in adata_input.varm and phi_key in adata_input.obsm:
            # Load the corresponding psi and phi initializations
            psi_init = torch.tensor(adata_input.varm[psi_key]).T  # Transpose for correct shape
            phi_init = torch.tensor(adata_input.obsm[phi_key])
            
            print(f"Shape of PSI_init is {psi_init.shape}")
            print(f"Shape of PHI_init is {phi_init.shape}")
        else:
            raise ValueError(f"PSI and PHI initializations for {K} waypoints not found in adata.varm or adata.obsm.")
    else:
        print(f"Random initialization of variational parameters!")
        psi_init = None 
        phi_init = None
    
    # Only empty CUDA cache if using a GPU
    if device.type == 'cuda':
        torch.cuda.empty_cache()

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

def compute_and_save_entropy_plots(assign_post, output_dir, report_file, plot=True, plot_filename='entropy_histogram.png'):
    """
    Computes cell-level entropy, overall entropy metrics, and optionally saves a histogram plot of entropies.

    Args:
        assign_post: The latent space matrix (cells by states) representing the probability distributions of cell state assignments.
        output_dir: Directory where to save the plot if plot=True.
        report_file: The file path where to log the entropy metrics.
        plot: Whether to save a histogram plot of cell-level entropies (default: True).
        plot_filename: Name of the histogram plot file (default: 'entropy_histogram.png').

    Returns:
        cell_entropies: Numpy array of entropy values for each cell.
        overall_mean_entropy: The mean entropy across all cells.
        overall_total_entropy: The total entropy across all cells.
    """
    
    # Step 1: Compute entropy for each cell (cell-level entropy)
    cell_entropies = entropy(assign_post, axis=1)  # Entropy for each row (cell)

    # Step 2: Compute overall entropy metrics
    overall_mean_entropy = np.mean(cell_entropies)  # Mean entropy across all cells
    overall_total_entropy = np.sum(cell_entropies)  # Sum of entropies (total uncertainty)
    
    # Log the results to the report file
    log_to_report(report_file, f"Overall mean entropy: {overall_mean_entropy}")
    log_to_report(report_file, f"Overall total entropy: {overall_total_entropy}")
    
    # Step 3: Optionally plot the histogram of cell-level entropies
    if plot:
        plt.figure(figsize=(8, 6))
        plt.hist(cell_entropies, bins=50, color='blue', alpha=0.7, edgecolor='black')
        plt.title("Histogram of Cell-Level Entropies", fontsize=16)
        plt.xlabel("Entropy", fontsize=14)
        plt.ylabel("Number of Cells", fontsize=14)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()

    return cell_entropies, overall_mean_entropy, overall_total_entropy

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

def plot_umap(latent_space, adata_input, output_dir, plot_name='umap.png', silhouette=None, dbi=None):
    """Plots UMAP with cell types, saves the plot, and computes both Silhouette Score and Davies-Bouldin Index."""
    
    # Create cell type colors
    cell_types_series = pd.Series(adata_input.obs.cell_type.astype(str).values)
    cell_types_unique = np.unique(cell_types_series)
    cell_type_colors = sns.color_palette("tab20", len(cell_types_unique))
    cell_type_dict = dict(zip(cell_types_unique, cell_type_colors))
    cell_colors = cell_types_series.map(cell_type_dict).values
    
    # Create the UMAP plot
    plt.figure(figsize=(6, 6))
    
    # Compute the silhouette score if not provided
    if silhouette is None:
        silhouette = silhouette_score(latent_space, adata_input.obs.cell_type.values, metric='euclidean')
    
    # Compute the DBI score if not provided
    if dbi is None:
        dbi = davies_bouldin_score(latent_space, adata_input.obs.cell_type.values)
    
    # Run UMAP first on embedding! 
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(latent_space)
    
    # Scatter plot for UMAP
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=cell_colors, alpha=0.5)
    
    # Display both metrics on the plot
    plt.text(0.05, 0.95, f'Silhouette Score: {silhouette:.2f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.05, 0.90, f'Davies-Bouldin Index: {dbi:.2f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top')
        
    # Add legend
    legend_handles = [mpatches.Patch(color=color, label=cell_type) for cell_type, color in cell_type_dict.items()]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, plot_name), bbox_inches='tight')
    plt.close()

    return silhouette, dbi, embedding

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

def compute_and_plot_albf(psis_mus, psis_loc, psis, pi, output_dir):
    
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

    # Save ALBF histogram plot to output directory
    plt.figure(figsize=(8, 6))
    plt.hist(albf_df["ALBF"], bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Histogram of ALBF values", fontsize=16)
    plt.xlabel("ALBF", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True)

    # Save the plot
    histogram_path = os.path.join(output_dir, 'albf_histogram.png')
    plt.savefig(histogram_path)
    plt.close()

    # Run All-vs-All differential splicing test
    print("Running All-vs-All differential splicing test...")
    all_vs_all_df = differential_splicing.all_vs_all_differential_splicing_test(psis_mus, psis_loc, pi)
    print(all_vs_all_df.head())

    # Save All-vs-All test results to a CSV file
    all_vs_all_output_path = os.path.join(output_dir, 'all_vs_all_results.csv')
    all_vs_all_df.to_csv(all_vs_all_output_path, index=False)

    print(f"All-vs-All differential splicing results saved to {all_vs_all_output_path}")

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

def prepare_output_directory(K_use, use_global_prior, input_conc, num_inits, num_epochs, cell_type_column, waypoint):
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
    output_dir = (f"./analysis_{timestamp}_K_{K_use}_{waypoint_str}_"
                  f"Prior_{use_global_prior_str}_NumEpochs_{num_epochs}_"
                  f"{input_conc_prior_str}_Inits_{num_inits}_"
                  f"{cell_type_str}_Random_{random_number}")
    
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
        "lr": args.lr,  # Learning rate
        "ATSE_file": args.ATSE_file,
        "run_NMF": args.run_NMF,
        "waypoints_use": args.waypoints_use,
        "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    # Save the parameters as a JSON file
    json_path = os.path.join(output_dir, 'parameters.json')
    with open(json_path, 'w') as json_file:
        json.dump(parameters, json_file, indent=4)
    
    print(f"Parameters saved to {json_path}")

def save_albf_to_csv(albf_df, output_dir):
    """Saves ALBF DataFrame to a CSV file in the output directory."""
    albf_path = os.path.join(output_dir, 'albf_results.csv')
    albf_df.to_csv(albf_path, index=False)
    print(f"ALBF results saved to {albf_path}")

def plot_umap_by_category(umap_coords, adata, category, output_dir, title, filename):
    """Plots UMAP coordinates colored by a categorical variable and saves the plot."""
    # Randomize cell indices
    n_cells = umap_coords.shape[0]
    random_indices = np.random.permutation(n_cells)  # Shuffle the indices

    # Convert categorical data to categorical values for coloring
    category_data = pd.Categorical(adata.obs[category])  # Keep the actual categories, not just codes
    codes = category_data.codes  # Get numerical codes for categories

    # Get a colormap with the number of unique categories
    n_categories = len(category_data.categories)
    colors = plt.cm.get_cmap('tab20', n_categories)  # Create colormap with distinct colors for each category

    # Increase overall font size
    plt.rcParams.update({'font.size': 18})

    # Plot UMAP, coloring by the selected category with shuffled order
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(umap_coords[random_indices, 0], umap_coords[random_indices, 1], 
                     c=codes[random_indices], cmap=colors, s=6, alpha=0.8)

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(title)

    # Create a discrete legend with unique category names
    unique_categories = category_data.categories  # Get the unique category names
    handles = [plt.Line2D([0], [0], marker='o', color=colors(i), lw=0, markersize=10) 
               for i in range(n_categories)]  # Create handles for each category
    plt.legend(handles, unique_categories, title=category, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the plot
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved UMAP plot colored by {category} to {plot_path}")

def prune_factors(pi, assign_post, psis_mus, psis_loc, psi_learned, threshold=0.005):
    """
    Prunes the latent variables based on the threshold for pi and renormalizes assign_post.
    """

    original_K = pi.shape[0]  # The original number of factors K

    # Step 1: Identify the factors with pi > threshold
    factor_threshold = threshold
    pruned_indices = np.where(pi > factor_threshold)[0]  # Indices where pi is greater than threshold
    pruned_K = pruned_indices.shape[0]  # The number of factors that remain after pruning

    # Print how many factors were originally and how many are retained
    print(f"Original number of factors (K): {original_K}")
    print(f"Number of factors retained after pruning: {pruned_K}")

    # Step 2: Prune latent variables based on the pruned indices
    pruned_pi = pi[pruned_indices]  # Pruned pi values
    # Renormalize pruned_pi so it sums to 1
    pruned_pi = pruned_pi / pruned_pi.sum()

    pruned_psis_mus = psis_mus[pruned_indices, :]  # Pruned psis_mus (only keep pruned factors)
    pruned_psis_loc = psis_loc[pruned_indices, :]  # Pruned psis_loc (only keep pruned factors)
    pruned_psi_learned = psi_learned[pruned_indices, :]  # Pruned psi_learned (only keep pruned factors)

    # Step 3: Prune assign_post and renormalize
    pruned_assign_post = assign_post[:, pruned_indices]  # Keep only the pruned factors in assign_post

    # Renormalize each row of assign_post so that it sums to 1
    pruned_assign_post = pruned_assign_post / pruned_assign_post.sum(axis=1, keepdims=True)

    return pruned_pi, pruned_assign_post, pruned_psis_mus, pruned_psis_loc, pruned_psi_learned

def multinomial_logistic_regression(adata_input, assign_post, K, output_dir, feature="cell_ontology_class"):
    """Performs multinomial logistic regression using latent factors as predictors and a categorical feature as the target."""
    
    # Create a DataFrame with latent factors from assign_post
    latent_df = pd.DataFrame(assign_post, columns=[f'Factor_{i}' for i in range(0, K)])
    
    # Ensure adata_input.obs is ordered by 'cell_id_index' and reset its index
    adata_input.obs = adata_input.obs.sort_values(by='cell_id_index').reset_index(drop=True)
    
    # Combine latent factors with adata.obs
    data_combined = pd.concat([adata_input.obs.reset_index(drop=True), latent_df], axis=1)

    # Set up data for regression model
    latent_factors = [f'Factor_{i}' for i in range(0, K)]
    X = data_combined[latent_factors]

    # Encode the target variable (e.g., 'sex', 'age')
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(data_combined[feature])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train a multinomial (or binary) logistic regression model
    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    logreg.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = logreg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{feature.capitalize()} Prediction Accuracy: {accuracy}")

    # Train the model on the full dataset for coefficient analysis
    logreg.fit(X, y_encoded)

    # Handle binary classification (e.g., 'sex') with special case
    if len(label_encoder.classes_) == 2:
        # Binary classification: Logistic regression gives one set of coefficients
        coefficients = pd.DataFrame(logreg.coef_, columns=latent_factors)
        coefficients.index = [f'Class: {label_encoder.classes_[1]} vs {label_encoder.classes_[0]}']
    else:
        # Multiclass case: One set of coefficients for each class
        coefficients = pd.DataFrame(logreg.coef_, columns=latent_factors, index=label_encoder.classes_)
        # Create a clustermap of the coefficients and save it as a figure
        sns.clustermap(coefficients, cmap="viridis", annot=False, fmt=".2f", figsize=(12, 10))

        # Save the figure to the output directory
        plot_filename = os.path.join(output_dir, f'multinomial_logreg_{feature}_coefficients.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"Clustermap saved to {plot_filename}")

    # Return accuracy score for logging or saving
    return accuracy

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

    # Read in the intron cluster file 
    print("Reading in obtained intron cluster (ATSE file!)")
    intron_clusts = pd.read_csv(ATSE_file, sep="}")
    genes = intron_clusts[["gene_id", "gene_name"]].drop_duplicates()

    if brain_only:
        print(f"Running model only on brain cells!")
    else:
        print(f"Running on all tissues!")

    # Prepare the output directory with timestamp
    output_dir = prepare_output_directory(K_use, use_global_prior, input_conc, num_inits, num_epochs, cell_type_column, waypoints_use)
    report_file = create_report_file(output_dir)

    # Load data and set K
    log_to_report(report_file, f"Loading anndata file from {input_path}")
    adata = load_adata(input_path)
    adata.var = pd.merge(adata.var, genes[['gene_id', 'gene_name']], how='left', on='gene_id')

    # Load sparse tensor model input files (need to improve this input processing)
    log_to_report(report_file, f"Loading sparse torch tensors as inputs to the model!")

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

    K = K_use

    # Save run parameters as JSON
    save_parameters(output_dir, args, K, input_conc)

    # Run the factor model
    log_to_report(report_file, f"Running factor model with K = {K}!")
    all_results, all_params = run_factor_model(adata, full_y_tensor, full_total_counts_tensor, K, device, use_global_prior, input_conc, waypoints_use, output_dir, num_inits, num_epochs, lr)
    save_plots(output_dir, all_results, report_file)

    # Calculate and plot correlations of assignment matrices
    log_to_report(report_file, "Calculating correlations between initializations.")
    assign_matrices = [result[0]["summary_stats"]["assign"]["mean"] for result in all_results]
    avg_corr, median_corr, min_corr = calculate_and_plot_correlations(assign_matrices, output_dir, report_file)

    # Select the best initialization based on the loss
    best_init = np.argmin([result[0]["losses"][-1] for result in all_results])
    latent_vars = all_results[best_init][0]['summary_stats']
    best_elbo = all_results[best_init][0]["losses"][-1]
    
    psis_mus = all_params[best_init]["AutoGuideList.0.loc"].reshape(K, latent_vars["psi"]["mean"].shape[1])
    psis_loc = all_params[best_init]["AutoGuideList.0.scale"].reshape(K, latent_vars["psi"]["mean"].shape[1])
    pi = latent_vars["pi"]["mean"]
    assign_post = latent_vars["assign"]["mean"]
    psi_learned = latent_vars["psi"]["mean"]
    a=latent_vars["a"]["mean"]
    b=latent_vars["b"]["mean"]

    input_conc_provided = input_conc

    # If input_conc was None extract it from latent variables 
    if input_conc is None:
        input_conc = latent_vars["bb_conc"]["mean"]
    else:
        input_conc = "infinity"

    log_to_report(report_file, f"Prunning factors!")
    pruned_pi, pruned_assign_post, pruned_psis_mus, pruned_psis_loc, pruned_psi_learned = prune_factors(pi, assign_post, psis_mus, psis_loc, psi_learned, threshold=0.005)
    new_K = len(pruned_pi)

    # Look at cell-level entropies based on latent assignments 
    log_to_report(report_file, "Calculating cell entropy given cell state assignments...")
    cell_entrops, mean_entropy, overall_total_entropy = compute_and_save_entropy_plots(pruned_assign_post, output_dir, report_file)
    adata.obs["cell_assignment_entropy"] = cell_entrops

    adata.var["junction_id_index"] = adata.var.index
    adata.var["junction_id_index"] = adata.var["junction_id_index"].astype(int)
    adata.var = adata.var.reset_index(drop=True)

    # Get metrics from ALBF
    log_to_report(report_file, "Running differential splicing analysis.")
    psis_df = compute_and_plot_albf(pruned_psis_mus, pruned_psis_loc, pruned_psi_learned, pruned_pi, output_dir)
    psis_df.rename(columns={i: f"cell_state_{i}" for i in range(new_K)}, inplace=True)
    # Save ALBF DataFrame
    save_albf_to_csv(psis_df, output_dir)
    
    adata.var = pd.concat([adata.var, psis_df.set_index('junction_id_index')], axis=1)    
    adata.obs["cell_type"] = adata.obs[cell_type_column]

    # Calculate silhouette score
    silhouette_avg, dbi, embedding = plot_umap(pruned_assign_post, adata, output_dir)
    plot_pi_barplot(pruned_pi, output_dir)
    plot_clustermap(pruned_assign_post, adata, output_dir)
    print(silhouette_avg, dbi)

    plot_umap_by_category(embedding, adata, "cell_type_grouped", output_dir, "UMAP colored by cell_type_grouped", "umap_cell_type_grouped.png")
    plot_umap_by_category(embedding, adata, "subtissue_clean", output_dir, "UMAP colored by Subtissue", "umap_subtissue.png")
    plot_umap_by_category(embedding, adata, "age", output_dir, "UMAP colored by Age", "umap_age.png")
    plot_umap_by_category(embedding, adata, "mouse.id", output_dir, "UMAP colored by Mouse ID", "umap_mouse.png")
    plot_umap_by_category(embedding, adata, "sex", output_dir, "UMAP colored by sex", "sex_mouse.png")

    log_to_report(report_file, "Running multinomial logistic regression using learned PHI and known cell annotations.")
    age_accuracy = multinomial_logistic_regression(adata, pruned_assign_post, new_K, output_dir, 'age')
    
    # Log accuracy or save it to a file
    print(f"Multinomial Logistic Regression Accuracy: {age_accuracy}")
    subtissue_accuracy = multinomial_logistic_regression(adata, pruned_assign_post, new_K, output_dir, 'subtissue_clean')
    
    # Log accuracy or save it to a file
    print(f"Multinomial Logistic Regression Accuracy: {subtissue_accuracy}")
    cell_type_accuracy = multinomial_logistic_regression(adata, pruned_assign_post, new_K, output_dir, 'cell_ontology_class')
    
    # Log accuracy or save it to a file
    print(f"Multinomial Logistic Regression Accuracy: {cell_type_accuracy}")
    
    # Perform logistic regression for mouse.id
    mouse_id_accuracy = multinomial_logistic_regression(adata, pruned_assign_post, new_K, output_dir, 'mouse.id')
    print(f"Multinomial Logistic Regression Accuracy for Mouse ID: {mouse_id_accuracy}")

    # Perform logistic regression for sex
    sex_accuracy = multinomial_logistic_regression(adata, pruned_assign_post, new_K, output_dir, 'sex')
    print(f"Multinomial Logistic Regression Accuracy for Sex: {sex_accuracy}")

    # Get silhouette score from NMF 
    # silhouette_NMF = get_NMF(adata, K=K, output_dir=output_dir)
    silhouette_NMF = "NA" 

    if waypoints_use:
        wayp = "YES"
    else:
        wayp = "NO"

    # Step 1: Define the columns for the DataFrame
    columns = [
        "K", "new_K", "use_global_prior", "learning_rate", 
        "avg_corr", "median_corr", "min_corr", 
        "silhouette_avg", "dbi", "cell_type_column", "silhouette_NMF", "learned_conc", "input_conc_provided", 
        "best_elbo", "cell_type_accuracy", "subtissue_accuracy", "age_accuracy", "mouse_id_accuracy", "sex_accuracy", "wayp", 
        "mean_cell_entropy", "overall_cell_total_entropy"]

    if cell_type_column == None:
        cell_type_column = "None"

    # Step 2: Create a DataFrame with a single row, add to this --> ELBO and 
    results_df = pd.DataFrame([[
        K, new_K, use_global_prior, lr, 
        avg_corr, median_corr, min_corr, 
        silhouette_avg, dbi, cell_type_column, silhouette_NMF, input_conc, input_conc_provided, best_elbo, 
        cell_type_accuracy, subtissue_accuracy, age_accuracy, mouse_id_accuracy, sex_accuracy, wayp, 
        mean_entropy, overall_total_entropy
    ]], columns=columns)

    # Step 3: Save the DataFrame to a CSV file in the output directory
    results_path = os.path.join(output_dir, "final_results.csv")
    results_df.to_csv(results_path, index=False)

    # Save all latent variables (i.e., trained model parameters) to a pickle file
    latent_vars_output_path = os.path.join(output_dir, "latent_vars.pkl")
    with open(latent_vars_output_path, "wb") as f:
        pickle.dump(latent_vars, f)
    log_to_report(report_file, f"Latent variables saved to {latent_vars_output_path}")
    
    # Save pruned latent variables 
    pruned_latent_variables = {
        "pruned_pi": pruned_pi,
        "pruned_assign_post": pruned_assign_post,
        "pruned_psis_mus": pruned_psis_mus,
        "pruned_psis_loc": pruned_psis_loc,
        "pruned_psi_learned": pruned_psi_learned,
        "new_K": new_K}
    
    latent_vars_output_path = os.path.join(output_dir, "pruned_latent_vars.pkl")
    with open(latent_vars_output_path, "wb") as f:
        pickle.dump(pruned_latent_variables, f)
    log_to_report(report_file, f"Pruned latent variables saved to {latent_vars_output_path}")

    print(f"Results saved to {results_path}")

    report_file.close()

if __name__ == "__main__":
    main()