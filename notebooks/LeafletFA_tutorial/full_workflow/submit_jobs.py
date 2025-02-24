import os
import json
import datetime

# Define where to save outputs 
# Should be directory in which model params are saved
base_output_dir = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/leafletFAmodel/2025-02-21/"

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
#SBATCH --job-name=leaflet_{job_id}
#SBATCH --output={log_dir}/leaflet_{job_id}.out
#SBATCH --error={log_dir}/leaflet_{job_id}.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=600G
#SBATCH --partition=bigmem

# Set Python to run in unbuffered mode to ensure real-time output
export PYTHONUNBUFFERED=1

# Run Python script with proper output handling
python -u /gpfs/commons/home/kisaev/Leaflet-private/notebooks/LeafletFA_tutorial/full_workflow/run_leaflet.py {param_id} 2>&1

# Optional: Add timestamp at the end of the job
echo "Job completed at: $(date)"
"""

# Run Python script
#python /gpfs/commons/home/kisaev/Leaflet-private/notebooks/LeafletFA_tutorial/run_leaflet.py {param_id}

# Generate and submit jobs
for i, params in enumerate(param_list):
    job_script = job_script_template.format(
        job_id=i,
        output_dir=base_output_dir,
        log_dir=log_dir,
        param_id=i
    )

    # Save job script
    job_file = os.path.join(base_output_dir, f"leaflet_job_{i}.slurm")
    with open(job_file, "w") as f:
        f.write(job_script)

    # Submit the job to Slurm
    os.system(f"sbatch {job_file}")

    # Remove the job script
    os.remove(job_file)

print(f"\nSubmitted {len(param_list)} jobs to Slurm.")
