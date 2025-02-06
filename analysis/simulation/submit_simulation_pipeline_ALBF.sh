#!/bin/bash

# Define possible values for each parameter
PROPORTION_NEGATIVE_VALUES=(0.5 0.1 0.9)
K_USE_VALUES=(2)
USE_GLOBAL_PRIOR_VALUES=(False True)
INPUT_CONC_PRIOR_VALUES=("None" "inf")  # Include "inf" as a string
CELL_TYPE_COLUMN_VALUES=("None" "cell_ontology_class")  # Add None as an option
WAYPOINTS_USE_VALUES=(False)  # Include option for waypoints
BRAIN_ONLY_VALUES=(True)  # Add brain_only as an option
SAVE_ANNDATA_VALUES=(False)  

repeats=1  # Repeat each combination 2 times

max_count=100
num_epochs=500
lr=0.1

# Script path 
analysis_script=/gpfs/commons/home/kisaev/Leaflet-private/src/simulation/simulate_pipeline_wALBF.py

# Anndata file input file path 
input_file=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/recomb_paper/ATSE_Anndata_Object_with_initializations_brain_only_20241018_134852.h5ad
ATSE_file=$input_file #doesn't matter not actually using this

# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/Simulations/2025/manuscript_sim_analysis/0204

# Initialize a counter
count=0

# Iterate through all parameter combinations
for proportion_negative in "${PROPORTION_NEGATIVE_VALUES[@]}"; do
  for K_use in "${K_USE_VALUES[@]}"; do
    for use_global_prior in "${USE_GLOBAL_PRIOR_VALUES[@]}"; do
      for input_conc_prior in "${INPUT_CONC_PRIOR_VALUES[@]}"; do
        for cell_type_column in "${CELL_TYPE_COLUMN_VALUES[@]}"; do
          for waypoints_use in "${WAYPOINTS_USE_VALUES[@]}"; do
            for brain_only in "${BRAIN_ONLY_VALUES[@]}"; do
              for save_anndata in "${SAVE_ANNDATA_VALUES[@]}"; do

                # Repeat each combination 'n' times
                for repeat in $(seq 1 $repeats); do
                  # Increment the counter
                  ((count++))

                  # Check if the counter exceeds the maximum number of combinations
                  if [ $count -gt $max_count ]; then
                    echo "Reached max count ($max_count), stopping."
                    exit 0
                  fi

                  # Submit the SLURM job script
                  sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=sim_data_${count}_rep${repeat}
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load necessary modules or activate your environment
conda activate LeafletSC  # If using a virtual environment

python $analysis_script --input_path $input_file \
  --proportion_negative ${proportion_negative} \
  --ATSE_file $ATSE_file \
  --K_use ${K_use} \
  --input_conc_prior ${input_conc_prior} \
  --num_inits 2 \
  --lr ${lr} \
  --num_epochs ${num_epochs} \
  --cell_type_column ${cell_type_column} \
  $( [[ $use_global_prior == "True" ]] && echo "--use_global_prior" ) \
  $( [[ $waypoints_use == "True" ]] && echo "--waypoints_use" ) \
  $( [[ $brain_only == "True" ]] && echo "--brain_only" ) \
  $( [[ $save_anndata == "True" ]] && echo "--save_anndata" )
EOT

                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Submitted $count jobs."