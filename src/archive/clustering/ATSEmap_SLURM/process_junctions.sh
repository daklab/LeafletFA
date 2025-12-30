#!/bin/bash
#SBATCH --job-name=junction_proc
#SBATCH --output=logs/junction_%A_%a.out
#SBATCH --error=logs/junction_%A_%a.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-99%30
#SBATCH -p bigmem

SCRIPT_PATH=/gpfs/commons/home/kisaev/Leaflet-private/src/clustering/testing_slurm.py
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/junction_processing_20250128

# Create base directory with today's date
cd $WD

# Print debug information
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Will process file: chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt"

# Check if chunk file exists
if [ ! -f "chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt" ]; then
    echo "Error: Chunk file chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt does not exist"
    exit 1
fi

# Process chunk
python $SCRIPT_PATH \
    --mode process \
    --chunk-file "chunks/chunk_${SLURM_ARRAY_TASK_ID}.txt" \
    --output-dir results