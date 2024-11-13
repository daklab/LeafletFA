#!/bin/bash

# Define possible values for each parameter
K_USE_VALUES=(20 30 50 100 200)
USE_GLOBAL_PRIOR_VALUES=(False True)
INPUT_CONC_PRIOR_VALUES=("None" "inf")  # Include "inf" as a string
CELL_TYPE_COLUMN_VALUES=("cell_type_grouped")  # Add None as an option
WAYPOINTS_USE_VALUES=(False True)  # Include option for waypoints
RUN_NMF_VALUES=(False)  # Include option for running NMF
BRAIN_ONLY_VALUES=(False)  # Add brain_only as an option
max_count=200
num_epochs=300
lr=0.1
repeats=3  # Repeat each combination 3 times

# Script path 
analysis_script=/gpfs/commons/home/kisaev/Leaflet-private/src/beta-dirichlet-factor/full_leafletFA_pipeline_wALBF.py

# Anndata file input file path 
input_file=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSE_Anndata_Object_with_initializations_20241017_162621.h5ad
ATSE_file=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/tabula_senis_test_intron_clusters_50_500000_10_20240927_single_cell.gz

# Initialize a counter
count=0

# Iterate through combinations
for K_use in "${K_USE_VALUES[@]}"; do
    for use_global_prior in "${USE_GLOBAL_PRIOR_VALUES[@]}"; do
        for input_conc_prior in "${INPUT_CONC_PRIOR_VALUES[@]}"; do
            for cell_type_column in "${CELL_TYPE_COLUMN_VALUES[@]}"; do
                for waypoints_use in "${WAYPOINTS_USE_VALUES[@]}"; do
                    for run_NMF in "${RUN_NMF_VALUES[@]}"; do
                        for brain_only in "${BRAIN_ONLY_VALUES[@]}"; do  # Add brain_only loop

                            # Repeat each combination 'n' times
                            for repeat in $(seq 1 $repeats); do

                                # Increment the counter
                                ((count++))

                                # Check if the counter exceeds the maximum number of combinations
                                if [ $count -gt $max_count ]; then
                                  echo "Reached max count ($max_count), stopping."
                                  exit 0
                                fi
                                
                                # Define the SLURM job script
                                sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${count}_ALL_${repeat}_real_data
#SBATCH --time=5-00:00:00
#SBATCH --mem=200G                    

# Load necessary modules or activate your environment
conda activate LeafletSC  # If using a virtual environment

python $analysis_script --input_path $input_file \
  --ATSE_file $ATSE_file \
  --K_use ${K_use} \
  --input_conc_prior ${input_conc_prior} \
  --num_inits 2 \
  --lr ${lr} \
  --num_epochs ${num_epochs} \
  --cell_type_column ${cell_type_column} \
  $( [[ $use_global_prior == "True" ]] && echo "--use_global_prior" ) \
  $( [[ $waypoints_use == "True" ]] && echo "--waypoints_use" ) \
  $( [[ $run_NMF == "True" ]] && echo "--run_NMF" ) \
  $( [[ $brain_only == "True" ]] && echo "--brain_only" )

EOT

                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Submitted $count jobs."

