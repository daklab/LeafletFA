# %%
# Load libraries and set up environment
import os 
import sys
import importlib
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import gffutils
import anndata as ad
import scanpy as sc 
from tqdm import tqdm
import json 

# Ensure CUDA is available and if not use CPU
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device found")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

float_type = {"device": device, "dtype": torch.float}
if device.type == "cuda":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Set seed for reproducibility
torch.manual_seed(0)

# Configure plotting styles
sns.set_theme()
sc.set_figure_params(figsize=(7, 7), frameon=True, dpi=80, facecolor='white')

# Define module paths
src_path = "/gpfs/commons/home/kisaev/Leaflet-private/src/"

# Add to sys.path if not already present
if src_path not in sys.path:
    sys.path.append(src_path)

# Import custom modules
import BetaDirichletFactor.LeafletFA as LeafletFA
import BetaDirichletFactor.differential_splicing as ds
import BetaDirichletFactor.utils as utils
import visualization.IsovizPy as ja
import evaluations.cost_correlation_assign as cca

# Reload custom modules
importlib.reload(LeafletFA)
importlib.reload(ds)
importlib.reload(utils)

# Simulation source code
sys.path.append("/gpfs/commons/home/kisaev/Leaflet-private/src/simulation/")
import simulate_counts as sim 
importlib.reload(sim)


# %%
# Define base output directory
base_output_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/Simulations/2025/manuscript_sim_analysis/2025-02-19"

# Get parameter set ID from command line
param_id = int(sys.argv[1])

# Load parameters
param_file = os.path.join(base_output_dir, "parameter_combinations.json")
with open(param_file, "r") as f:
    param_list = json.load(f)
params = param_list[param_id] 

# Define output directory
output_dir = os.path.join(base_output_dir, f"run_{param_id}")
os.makedirs(output_dir, exist_ok=True)
print(f"All outputs will be saved in {output_dir}")

# %%
# Anndata file input file path 
ATSE_anndata_file="/gpfs/commons/groups/knowles_lab/Karin/TMS_MODELING/DATA_FILES/BRAIN_ONLY/02112025/TMS_Anndata_ATSE_counts_with_waypoints_20250211_171237.h5ad"
ATSE_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/ATSEfiles/TMS_atse_file_unanno_also_2025-01-30_19-24-18.txt.gz"

# %%
# Load splicing anndata file along with the ATSE annotation file (both obtained using upstream processing within LeafletFA framework)
adata = ad.read_h5ad(ATSE_anndata_file)
atses = pd.read_csv(ATSE_file, sep="\t")

# %%
# Filter adata to only include junctions that have non_zero_count_cells >= 10
adata = adata[:, adata.var["non_zero_count_cells"] > 2]

# choose which column should be used for maintaining cell labels when simulating data...
sim_label_column = params["sim_label_column"] #"cell_type_grouped" # or set to None then cells will be randomly assigned to groups
proportion_negative = params["proportion_negative"]

if params["sim_label_column"] is None:
    K = 2
else:
    K = len(adata.obs[sim_label_column].unique())

# Set up some useful params 
params["input_conc"] = None if params["input_conc"] is None else torch.tensor(np.inf)
input_conc = params["input_conc"]
junc_specific_prior = params["junc_specific_prior"] # set to True if you want to use a junc-specific prior (a set of a,b shape params for each junction) or False to learn one set of a,b shape params for all junctions
waypoints_use = params["waypoints_use"] # don't have waypoints in simulated data

# %%
# Preprocess the data
adata_filtered = sim.preprocess_adata(adata, sim_label_column, "cell_by_cluster_matrix")
# Simulate data
_, _, adata_input, cell_type_psi_df = sim.simulate_and_prepare_data(adata_filtered, K, float_type, proportion_negative, sim_label_column)

# Write cell_type_psi_df to a CSV file 
cell_type_psi_df_path = os.path.join(output_dir, 'cell_type_psi_df.csv')
cell_type_psi_df.to_csv(cell_type_psi_df_path, index=False)

# %% [markdown]
# ### Initialize and train the model using the simulated counts!

# %%
num_inits = params["num_inits"]
num_epochs = params["num_epochs"]
num_samples = params["num_samples"]
lr = params["lr"]
ELBO_num_particles = params["ELBO_num_particles"]
print_epochs = 10

# %%
# Let's initialize the LeafletFA class 
leaflet_model = LeafletFA.LeafletFA(adata=adata_input, K=K, 
                                    junc_specific_prior = junc_specific_prior, 
                                    waypoints_use=waypoints_use, 
                           input_conc_prior = input_conc, 
                           num_epochs=num_epochs, 
                           print_epochs=print_epochs, 
                           ELBO_num_particles=ELBO_num_particles, 
                           lr=lr, gamma=0.05, 
                           num_samples=num_samples, 
                           output_dir=output_dir)

# Convert AnnData into PyTorch tensors for model training 
leaflet_model.from_anndata()

# Train the model 
leaflet_model.train(num_initializations=num_inits)

# Get the best initialization and extract all the latent variables at this initialization
# If you want the latent variables from a different initialization, you can pass the index of 
# that initialization to the get_all_variables() function
leaflet_model.get_all_variables()

# %%
# let's look at the results 
assign_matrices = [result["summary_stats"]["assign"]["mean"] for result in leaflet_model.latent_results]

# Calculate correlations between initializations if more than 2 
if num_inits > 1:
    avg_corr, median_corr, min_corr = utils.calculate_and_plot_correlations(assign_matrices)
else: 
    avg_corr, median_corr, min_corr = None, None, None  # Set default values when there's only one initialization

# %%
# Prune K: note this updates all the latent variables in the model to only include estimates for the pruned K
leaflet_model.prune_K()

# %%
LEAFLETFA_LATENT_KEY = "X_leafletFA"
adata_input.obsm[LEAFLETFA_LATENT_KEY] = leaflet_model.assign_post # assign_post is the posterior assignment cell-factor activity matrix 
sc.pp.neighbors(adata_input, use_rep=LEAFLETFA_LATENT_KEY, n_neighbors=10)

# %%
# Compute UMAP
print(f"Computing UMAP for K={K}...")
sc.tl.umap(adata_input)

# Define UMAP save path
umap_save_path = os.path.join(output_dir, f"UMAP_K_{K}.png")

# Set figure parameters (size, dpi, font settings)
with plt.rc_context({'figure.figsize': (7, 7), 'savefig.dpi': 300}):  
    sc.pl.umap(
        adata_input, 
        color=["cell_type"], 
        show=False  # Don't show interactive plot
    )
    plt.savefig(umap_save_path, bbox_inches="tight")  # Save with tight bounding box
print(f"Saved UMAP plot to {umap_save_path}")

# %%
cell_tye_silhouette = ds.calculate_silhouette_score(leaflet_model.assign_post, adata_input.obs.cell_type.values)
print(f"Silhouette score for cell types: {cell_tye_silhouette}")

# %%
pi_df = pd.DataFrame(leaflet_model.pi, columns=["factor_assignment_probabilities"])
pi_df["factor_K"] = pi_df.index

# Sort by factor_assignment_probabilities in descending order
pi_df = pi_df.sort_values(by="factor_assignment_probabilities", ascending=False)

# Make sorted barplot
plt.figure(figsize=(10, 5))
sns.barplot(x="factor_K", y="factor_assignment_probabilities", data=pi_df, order=pi_df["factor_K"])
plt.title("Factor K Probabilities (Sorted)")
plt.xlabel("Factor K")
plt.ylabel("Assignment Probabilities")
plt.xticks(rotation=90)  # Rotate x-axis labels if many factors
plt.savefig(os.path.join(output_dir, "factor_assignment_probabilities.png"))

# %%
# Let's extract sampled PSI means to calculate imputed difference between groups
# convert leaflet_model.psi_samples to a dataframe rename columns to factor_
factor_psi_df = pd.DataFrame(leaflet_model.psi_learned.T)
factor_psi_df.columns = [f"factor_imputed_psi_{col}" for col in factor_psi_df.columns]

if K == 2:
    factor_psi_df["imputed_diff"] = np.abs(factor_psi_df["factor_imputed_psi_0"] - factor_psi_df["factor_imputed_psi_1"])
if K > 2:
    # Calculate variance across all factors
    factor_psi_df["imputed_diff"] = factor_psi_df.var(axis=1)

# Add factor_psi_df to adata_input.var 
adata_input.var = pd.concat([adata_input.var, factor_psi_df], axis=1)

# Calculate pearson and spearman correlation
pearson_corr = stats.pearsonr(adata_input.var["imputed_diff"], adata_input.var["difference"])[0]
spearman_corr = stats.spearmanr(adata_input.var["imputed_diff"], adata_input.var["difference"])[0]
print(f"Spearman correlation between imputed and true difference: {spearman_corr}")

# Plot scatterplot correlation between imputed difference and true difference 
plt.figure(figsize=(6, 6))
sns.scatterplot(x="imputed_diff", y="difference", data=adata_input.var)
plt.xlabel("Imputed Difference")
plt.ylabel("True Difference")

# Add rounded spearman_corr to title 
plt.title(f"Imputed vs True Difference in PSI (Spearman: {round(spearman_corr, 2)})")
# Save to output_dir
plt.savefig(os.path.join(output_dir, "imputed_vs_true_diff.png"))

# %%
if K > 2:
    results = ds.analyze_all_factors(leaflet_model, fdr_threshold=0.05, min_effect_size=0.1)

    # Let's collect all the dataframes from results
    # go through results.keys() and collect second value in tuple and add column indicating factor K
    results_df = pd.DataFrame()
    for key in results.keys():
        factor_res = results[key][1]
        factor_res["factor_K"] = key
        results_df = pd.concat([results_df, factor_res])

    # for every unique junctions i want to summarize number of factors that it was significant in
    sig_only = results_df[results_df["significant"] == True]
    sig_only = sig_only.groupby("junction_idx").agg({"significant": "count"}).reset_index()
    sig_only = sig_only.rename(columns={"significant": "num_factors_significant"})
    # merge with results_df
    results_df = results_df.merge(sig_only, on="junction_idx", how="left")
    results_df["abs_effect_size"] = np.abs(results_df["effect_size"])

    # plot dist of effect sizes
    plt.figure(figsize=(6, 6))
    sns.histplot(results_df["effect_size"], bins=30)
    plt.xlabel("Effect Size")
    plt.ylabel("Frequency")
    # save to output_dir
    plt.savefig(os.path.join(output_dir, "effect_size_dist.png"))

# %%
if K == 2:
    results_dict, results_df = ds.analyze_differential_splicing(leaflet_model, factor_idx = 0, 
                                                            fdr_threshold=0.05, 
                                                            min_effect_size=0.1)
                                        
    # Add results_df to adata_input.var
    adata_input.var = pd.concat([adata_input.var, results_df], axis=1)

    # Relabel positive true_label junctions as negative if their difference is less than 0.1 (same effect size we use later for differential splicing calling)
    adata_input.var["adjusted_true_label"] = adata_input.var["true_label"]
    adata_input.var.loc[(adata_input.var["true_label"] == "positive") & (adata_input.var["difference"] < 0.1), "adjusted_true_label"] = "negative"

    # Create confusion matrix directly from DataFrame
    conf_matrix = pd.crosstab(adata_input.var['adjusted_true_label'], adata_input.var['significant'], 
                             normalize=False)  # set to True if you want percentages

    # Plot
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted (significant)')
    plt.ylabel('True Label')
    # save to output_dir
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    
    # Get precision, recall, auc_pr
    precision, recall, auc_pr = ds.plot_precision_recall_curve(adata_input, results_df, output_dir=output_dir)

    # Get fpr, tpr, auc_roc
    fpr, tpr, auc_roc = ds.plot_roc_curve(adata_input, results_df, output_dir=output_dir)

    # add absolute effective size 
    adata_input.var["abs_effect_size"] = np.abs(adata_input.var["effect_size"])
    # Plot scatterplot correlation between imputed difference and true difference 
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x="abs_effect_size", y="difference", data=adata_input.var)
    plt.xlabel("DS Effect Size")
    plt.ylabel("True Estimated Difference")
    plt.title("Imputed vs True Difference in PSI")
    # save to output_dir
    plt.savefig(os.path.join(output_dir, "imputed_vs_true_diff.png"))

    # Calculate pearson and spearman correlation
    pearson_corr = stats.pearsonr(adata_input.var["abs_effect_size"], adata_input.var["difference"])[0]
    spearman_corr = stats.spearmanr(adata_input.var["abs_effect_size"], adata_input.var["difference"])[0]
    print(f"Pearson correlation between abs_effect_size and true difference: {pearson_corr}")
    print(f"Spearman correlation between abs_effect_size and true difference: {spearman_corr}")

    xlabel = "Estimated Effect Size B_j"
    ylabel = "Frequency"

    # color by adjusted_true_label
    plt.figure(figsize=(6, 6))
    sns.histplot(data=adata_input.var, x="effect_size", hue="adjusted_true_label")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # save to output_dir
    plt.savefig(os.path.join(output_dir, "effect_size_dist.png"))

    true_positive_junctions = set(adata_input.var[adata_input.var["adjusted_true_label"] == "positive"].index)
    df_fdr_results = ds.calibration_test(leaflet_model, true_positive_junctions, min_effect_size=0.1)

    # Save all these metrics and plots to output_dir
    # Add precision, recall, auc_pr, fpr, tpr, auc_roc to classification_metrics.csv
    classification_metrics = pd.DataFrame([{
        "auc_pr": auc_pr,
        "auc_roc": auc_roc
    }])

    classification_metrics_file = os.path.join(output_dir, "classification_metrics.csv")
    classification_metrics.to_csv(classification_metrics_file, index=False)

# %%
# Collect all params used in this analysis as well as the results

# Create a dataframe to store results
results_df = pd.DataFrame([{
    "param_id": param_id,
    "K": K,
    "junc_specific_prior": params["junc_specific_prior"],
    "best_elbo": leaflet_model.best_elbo,
    "input_conc": leaflet_model.bb_conc,
    "num_epochs": params["num_epochs"],
    "num_inits": params["num_inits"],
    "cell_type_silhouette": cell_tye_silhouette,
    "avg_corr": avg_corr,
    "median_corr": median_corr,
    "min_corr": min_corr,
    "lr": params["lr"],
    "ELBO_num_particles": params["ELBO_num_particles"],
    "num_samples": params["num_samples"],
    "proportion_negative": params["proportion_negative"],
    "pruned_K": len(leaflet_model.pi)  # Number of factors retained after pruning
}])

# Save results dataframe
results_file = os.path.join(output_dir, "run_summary.csv")
results_df.to_csv(results_file, index=False)

print(f"Saved run summary to {results_file}")

