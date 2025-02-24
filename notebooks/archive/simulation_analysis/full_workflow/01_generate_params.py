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

# Define parameter grid
param_grid = {
    "input_conc": [None, "inf"],  # 'inf' will be converted to torch.tensor(np.inf)
    "junc_specific_prior": [True, False],
    "sim_label_column": [None, "cell_type_grouped"],
    "waypoints_use": [False],
    "num_inits": [5],
    "num_samples": [500],
    "lr": [0.1, 0.5, 0.8],
    "proportion_negative": [0.5, 0.15, 0.85],
    "ELBO_num_particles": [10],
    "num_epochs": [300]
}

# Generate all parameter combinations
param_combinations = list(itertools.product(*param_grid.values()))

# Convert to list of dictionaries with replicate IDs (3 repeats per param set)
param_list = [
    {**dict(zip(param_grid.keys(), values)), "replicate_id": replicate}
    for values in param_combinations
    for replicate in range(3)  # Generate three replicates per combination
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
