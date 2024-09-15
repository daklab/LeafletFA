#!/bin/bash

# Define possible values for each parameter
K_USE_VALUES=(2 5 10 20 50 100)
USE_GLOBAL_PRIOR_VALUES=(False True)
INPUT_CONC_PRIOR_VALUES=("None" "inf")  # Include "inf" as a string
CELL_TYPE_COLUMN_VALUES=("cell_type")  # Add None as an option
max_count=100
num_epochs=1000
lr=0.2

# Script path 
analysis_script=/gpfs/commons/home/kisaev/Leaflet-private/src/beta-dirichlet-factor/full_leafletFA_pipeline_wALBF.py

# Anndata file input file path 
input_file=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leafletall_ages_brain_intron_clusters_adata.h5ad

# Initialize a counter
count=0

# Iterate through combinations
for K_use in "${K_USE_VALUES[@]}"; do
    for use_global_prior in "${USE_GLOBAL_PRIOR_VALUES[@]}"; do
        for input_conc_prior in "${INPUT_CONC_PRIOR_VALUES[@]}"; do
            for cell_type_column in "${CELL_TYPE_COLUMN_VALUES[@]}"; do

            # Increment the counter
            ((count++))

            # Check if the counter exceeds the maximum number of combinations
            if [ $count -gt $max_count ]; then
                break 4  # Break all loops if max_count is reached
            fi

            # Define the SLURM job script
            sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=real_data_${count}
#SBATCH --output=real_data_${count}.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

# Load necessary modules or activate your environment
conda activate LeafletSC  # If using a virtual environment

python $analysis_script --input_path $input_file \
  --K_use ${K_use} \
  --input_conc_prior ${input_conc_prior} \
  --num_inits 10 \
  --lr ${lr} \
  --num_epochs ${num_epochs} \
  $( [[ $use_global_prior == "True" ]] && echo "--use_global_prior" ) \
  $( [[ $cell_type_column != "None" ]] && echo "--cell_type_column ${cell_type_column}" )

EOT

        done
      done
    done
  done

echo "Submitted $count jobs."