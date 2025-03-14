# generate_params.py

import itertools
import json
import os
import datetime
import pandas as pd

# Define output directory
# Define base output directory
base_output_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/leafletFAmodel/"

# Create output directory if it doesn't exist with today's date inside base_output_dir
today = datetime.datetime.now().strftime("%Y-%m-%d")
base_output_dir = os.path.join(base_output_dir, today)
os.makedirs(base_output_dir, exist_ok=True)
print(f"All outputs will be saved in {base_output_dir}")

# Define parameter grid
param_grid = {
    "input_conc": [None],  # 'inf' will be converted to torch.tensor(np.inf)
    "junc_specific_prior": [True],
    "K": [30, 100],
    "waypoints_use": [True],
    "num_inits": [1],
    "ELBO_num_particles": [2],
    "num_samples": [100],
    'gamma': [0.05],
    'min_delta': [50],
    "lr": [0.9],
    "num_epochs": [5, 200],
    "patience": [3],
}

# Generate all parameter combinations
param_combinations = list(itertools.product(*param_grid.values()))

# Convert to list of dictionaries
param_list = [
    dict(zip(param_grid.keys(), values)) for values in param_combinations
]

# Save parameter combinations to JSON
param_file = os.path.join(base_output_dir, "parameter_combinations.json")
with open(param_file, "w") as f:
    json.dump(param_list, f, indent=4)

# Also save as a CSV
param_df = pd.DataFrame(param_list)
param_df.to_csv(os.path.join(base_output_dir, "parameter_combinations.csv"), index=False)

print(f"Generated {len(param_list)} parameter sets.")
print(f"Parameter JSON saved to: {param_file}")
print(f"Parameter CSV saved to: {os.path.join(base_output_dir, 'parameter_combinations.csv')}")
