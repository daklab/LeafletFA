# run_leaflet.py

# %%
# Load libraries and set up environment
import os 
import sys
import datetime
import numpy as np
import pandas as pd
import anndata as ad    
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc 
import json
import pickle
import lzma

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

# Define base output directory
base_output_dir = "/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LeafletFA/ATSEs/022025/output/LeafletFA_model/2025-02-23/"

# Get parameter set ID from command line
param_id = int(sys.argv[1])
print(f"Loading parameter set {param_id}...")

# Load parameters
param_file = os.path.join(base_output_dir, "parameter_combinations.json")
with open(param_file, "r") as f:
    param_list = json.load(f)
params = param_list[param_id] 

# Convert 'inf' string to torch.tensor(np.inf)
params["input_conc"] = None if params["input_conc"] is None else torch.tensor(np.inf)

# Define output directory
output_dir = os.path.join(base_output_dir, f"run_{param_id}")
os.makedirs(output_dir, exist_ok=True)
print(f"All outputs will be saved in {output_dir}")

# Load Anndata file
ATSE_anndata_file="/gpfs/commons/projects/knowles_singlecell_splicing/PRJEB14362/LeafletFA/ATSEs/022025/anndatas/iPSC_human_Anndata_ATSE_counts_with_waypoints_20250214_171021.h5ad"

print(f"Loading Anndata file: {ATSE_anndata_file}")
adata = ad.read_h5ad(ATSE_anndata_file)
print(f"Anndata file loaded successfully.")

# Initialize model
print("Initializing LeafletFA model...")
leaflet_model = LeafletFA.LeafletFA(
    adata=adata, 
    K=params["K"], 
    junc_specific_prior=params["junc_specific_prior"], 
    waypoints_use=params["waypoints_use"], 
    input_conc_prior=params["input_conc"], 
    num_epochs=params["num_epochs"], 
    print_epochs=1, 
    ELBO_num_particles=params["ELBO_num_particles"], 
    lr=params["lr"], 
    gamma=0.05, 
    num_samples=params["num_samples"], 
    output_dir=output_dir
)

# Train model
print("Training LeafletFA model...")
leaflet_model.from_anndata()
leaflet_model.train(num_initializations=params["num_inits"])

print("Training complete, extracting results...")
leaflet_model.get_all_variables()
# Prune K! 
print("Pruning K...")
leaflet_model.prune_K()

# Calculate correlations between initializations if more than 2 
assign_matrices = [result["summary_stats"]["assign"]["mean"] for result in leaflet_model.latent_results]
if params["num_inits"] > 1:
    avg_corr, median_corr, min_corr = utils.calculate_and_plot_correlations(assign_matrices)
else: 
    avg_corr, median_corr, min_corr = None, None, None  # Set default values when there's only one initialization

# Save latent variables
adata.obsm[f"X_leafletFA_K{params['K']}"] = leaflet_model.assign_post

cell_tye_silhouette = ds.calculate_silhouette_score(leaflet_model.assign_post, adata.obs.day.values)

print(f"Silhouette score for day: {cell_tye_silhouette}")

# Compute UMAP
print(f"Computing UMAP for K={params['K']}...")
sc.pp.neighbors(adata, use_rep=f"X_leafletFA_K{params['K']}")
sc.tl.umap(adata)

# Define UMAP save path
umap_save_path = os.path.join(output_dir, f"UMAP_K{params['K']}.png")

# Set figure parameters (size, dpi, font settings)
with plt.rc_context({'figure.figsize': (10, 7), 'savefig.dpi': 300}):  
    sc.pl.umap(
        adata, 
        color=["day"], 
        wspace=0.9, 
        show=False  # Don't show interactive plot
    )
    plt.savefig(umap_save_path, bbox_inches="tight")  # Save with tight bounding box

print(f"Saved UMAP plot to {umap_save_path}")

pi_df = pd.DataFrame(leaflet_model.pi, columns=["factor_assignment_probabilities"])
pi_df["factor_K"] = pi_df.index+1
pi_df.to_csv(os.path.join(output_dir, "factor_assignment_probabilities.csv"), index=False)
print(f"Saved factor assignment probabilities to {os.path.join(output_dir, 'factor_assignment_probabilities.csv')}")

# Create a dataframe to store results
results_df = pd.DataFrame([{
    "param_id": param_id,
    "K": params["K"],
    "junc_specific_prior": params["junc_specific_prior"],
    "waypoints_use": params["waypoints_use"],
    "best_elbo": leaflet_model.best_elbo,
    "input_conc": leaflet_model.bb_conc,
    "num_epochs": params["num_epochs"],
    "num_inits": params["num_inits"],
    "cell_type_silhouette": cell_tye_silhouette,
    "avg_corr": avg_corr,
    "median_corr": median_corr,
    "min_corr": min_corr,
    "lr": params["lr"],
    "num_samples": params["num_samples"],
    "ELBO_num_particles": params["ELBO_num_particles"],
    "pruned_K": len(leaflet_model.pi)  # Number of factors retained after pruning
}])

# Save results dataframe
results_file = os.path.join(output_dir, "run_summary.csv")
results_df.to_csv(results_file, index=False)
print(f"Saved run summary to {results_file}")

# Save leafletfa model object 
model_file = os.path.join(output_dir, "leafletfa_model.pkl.xz")
# Save the trained LeafletFA model (without the adata object)
leaflet_model.adata = None
# Remove tensors also 
leaflet_model.y = None
leaflet_model.total_counts = None
print(f"Saving model to {model_file}")

with lzma.open(model_file, "wb") as f:
    pickle.dump(leaflet_model, f)

print(f"Model saved to {model_file}")