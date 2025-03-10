import os
import json

# Define output directory
base_output_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/Simulations/2025/manuscript_sim_analysis/2025-02-22"

# Load parameter list from JSON file
param_file = os.path.join(base_output_dir, "parameter_combinations.json")
if not os.path.exists(param_file):
    raise FileNotFoundError(f"Parameter file {param_file} not found. Run generate_params.py first!")

with open(param_file, "r") as f:
    param_list = json.load(f)

# Create logs directory
log_dir = os.path.join(base_output_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

# Slurm job script template
job_script_template = """#!/bin/bash
#SBATCH --job-name=SIMLEAF_{job_id}
#SBATCH --output={log_dir}/leafletSIM_{job_id}.out
#SBATCH --error={log_dir}/leafletSIM_{job_id}.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Run Python script
python /gpfs/commons/home/kisaev/Leaflet-private/notebooks/archive/simulation_analysis/full_workflow/02_run_simulation_and_leaflet.py {param_id}
"""

# Generate and submit jobs
for i, params in enumerate(param_list):
    job_id = f"{i}_rep{params['replicate_id']}"  # Unique ID with replicate number

    job_script = job_script_template.format(
        job_id=job_id,
        log_dir=log_dir,
        param_id=i
    )

    # Save job script
    job_file = os.path.join(base_output_dir, f"leaflet_job_{job_id}.slurm")
    with open(job_file, "w") as f:
        f.write(job_script)

    # Submit the job to Slurm
    os.system(f"sbatch {job_file}")

    # Remove the job script after submission
    os.remove(job_file)

print(f"Submitted {len(param_list)} jobs to Slurm.")
