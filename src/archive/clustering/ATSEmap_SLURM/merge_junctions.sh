#!/bin/bash
# merge_junctions.sh
#SBATCH --job-name=junction_merge
#SBATCH --output=logs/junction_merge_%j.out
#SBATCH --error=logs/junction_merge_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-private/src/clustering/testing_slurm.py
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/junction_processing_20250128

# Create base directory with today's date
cd $WD

# Merge results
python $SCRIPT_PATH \
    --mode merge \
    --output-dir results \
    --merge-output results/final_junctions.pkl