import itertools
import json
import os
import datetime
import pandas as pd

# Define base output directory
base_output_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/Simulations/2025/manuscript_sim_analysis"

# Create output directory with today's date
today = datetime.datetime.now().strftime("%Y-%m-%d")
base_output_dir = os.path.join(base_output_dir, today)
os.makedirs(base_output_dir, exist_ok=True)
print(f"All outputs will be saved in {base_output_dir}")

# Define concise parameter grid (only unique values)
param_grid = {
    "input_conc": [None, "inf"],  # You can convert "inf" to np.inf later
    "junc_specific_prior": [True, False],
    "delta_fixed": [0.5, 1, None],
    "sim_label_column": [None, "cell_type_grouped"],
    "waypoints_use": [False],
    "num_inits": [5],
    "num_samples": [100],
    "lr": [0.2, 0.8],
    "gamma": [0.01, 0.05],
    "proportion_negative": [0.5, 0.2, 0.8],
    "ELBO_num_particles": [10],
    "num_epochs": [200],
    "device": ["cpu"]
}

# Generate all combinations of unique parameter settings
param_combinations = list(itertools.product(*param_grid.values()))

# Create param dicts with replicate IDs
param_list = [
    {**dict(zip(param_grid.keys(), values)), "replicate_id": replicate}
    for values in param_combinations
    for replicate in range(1)
]

# Save parameter combinations to JSON
param_file = os.path.join(base_output_dir, "parameter_combinations.json")
with open(param_file, "w") as f:
    json.dump(param_list, f, indent=4)

# Also save as a CSV
param_df = pd.DataFrame(param_list)
param_df.to_csv(os.path.join(base_output_dir, "parameter_combinations.csv"), index=False)

print(f"Generated {len(param_list)} parameter sets (including replicates).")
print(f"Parameter JSON saved to: {param_file}")
print(f"Parameter CSV saved to: {os.path.join(base_output_dir, 'parameter_combinations.csv')}")
