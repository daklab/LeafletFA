#!/bin/bash

# Define possible values for each parameter
K_USE_VALUES=(20 100)
USE_GLOBAL_PRIOR_VALUES=(False True)
INPUT_CONC_PRIOR_VALUES=("None" "inf")  # Include "inf" as a string
CELL_TYPE_COLUMN_VALUES=("cell_type")
MASKS=(0.01 0.5 0.9999)
max_count=2
num_epochs=40
lr=0.01

# Script path 
analysis_script=/gpfs/commons/home/kisaev/Leaflet-private/src/evaluations/masking_BBFactor.py

# Anndata file input file path 
input_file=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leafletall_ages_brain_intron_clusters_adata.h5ad

# Initialize a counter
count=0

# Iterate through combinations
for mask_perc in "${MASKS[@]}"; do
  for K_use in "${K_USE_VALUES[@]}"; do
    for use_global_prior in "${USE_GLOBAL_PRIOR_VALUES[@]}"; do
      for input_conc_prior in "${INPUT_CONC_PRIOR_VALUES[@]}"; do
        for cell_type_column in "${CELL_TYPE_COLUMN_VALUES[@]}"; do

          # Increment the counter
          ((count++))

          # Check if the counter exceeds the maximum number of combinations
          if [ $count -gt $max_count ]; then
            break
          fi

          # Define the SLURM job script
          sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=job_${count}
#SBATCH --output=output_${count}.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

# Load necessary modules or activate your environment
conda activate LeafletSC 

python $analysis_script --input_path $input_file \
  --mask_perc ${mask_perc} \
  --K_use ${K_use} \
  --input_conc_prior ${input_conc_prior} \
  --num_inits 3 \
  --lr ${lr} \
  --num_epochs ${num_epochs} \
  $( [[ $use_global_prior == "True" ]] && echo "--use_global_prior" ) \
  $( [[ $cell_type_column != "None" ]] && echo "--cell_type_column ${cell_type_column}" )

EOT

        done
        # Break out of the loops if 10 combinations have been submitted
        if [ $count -gt $max_count ]; then
          break
        fi
      done
      if [ $count -gt $max_count ]; then
        break
      fi
    done
    if [ $count -gt $max_count ]; then
      break
    fi
  done
  if [ $count -gt $max_count ]; then
    break
  fi
done

echo "Submitted $count jobs."
