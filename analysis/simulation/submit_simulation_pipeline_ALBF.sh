#!/bin/bash

# Define possible values for each parameter
PROPORTION_NEGATIVE_VALUES=(0.1 0.5 0.9)
K_USE_VALUES=(2)
USE_GLOBAL_PRIOR_VALUES=(True False)
INPUT_CONC_PRIOR_VALUES=("None" "inf")  # Include "inf" as a string
CELL_TYPE_COLUMN_VALUES=("None" "cell_type")  # Add None as an option
max_count=300
num_epochs=600
lr=0.05

# Script path 
analysis_script=/gpfs/commons/home/kisaev/Leaflet-private/src/simulation/simulate_pipeline_wALBF.py

# Anndata file input file path 
input_file=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leafletall_ages_brain_intron_clusters_adata.h5ad

# Initialize a counter
count=0

# Iterate through combinations
for proportion_negative in "${PROPORTION_NEGATIVE_VALUES[@]}"; do
  for K_use in "${K_USE_VALUES[@]}"; do
    for use_global_prior in "${USE_GLOBAL_PRIOR_VALUES[@]}"; do
      for input_conc_prior in "${INPUT_CONC_PRIOR_VALUES[@]}"; do
        for cell_type_column in "${CELL_TYPE_COLUMN_VALUES[@]}"; do
          
          # Run each combination three times
          for repeat in {1..3}; do
          
            # Increment the counter
            ((count++))

            # Check if the counter exceeds the maximum number of combinations
            if [ $count -gt $max_count ]; then
              break
            fi

            # Define the SLURM job script
            sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=sim_data_${count}_rep${repeat}
#SBATCH --output=sim_data_${count}_rep${repeat}.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

# Load necessary modules or activate your environment
conda activate LeafletSC  # If using a virtual environment

python $analysis_script --input_path $input_file \
  --proportion_negative ${proportion_negative} \
  --K_use ${K_use} \
  --input_conc_prior ${input_conc_prior} \
  --num_inits 3 \
  --lr ${lr} \
  --num_epochs ${num_epochs} \
  $( [[ $use_global_prior == "True" ]] && echo "--use_global_prior" ) \
  $( [[ $cell_type_column != "None" ]] && echo "--cell_type_column ${cell_type_column}" )

EOT

          done
          
          # Break out of the loops if max_count combinations have been submitted
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
  if [ $count -gt $max_count ]; then
    break
  fi
done

echo "Submitted $count jobs."
