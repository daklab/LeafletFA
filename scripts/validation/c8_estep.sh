#!/usr/bin/env bash
# Check 8 E-step (GPU): training cells against the frozen consensus fit; K read from the consensus fit.
#   sbatch -J lfa_c8_estep -c 8 --mem 12G -t 1:00:00 c8_estep.sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# ne1dg7 (b6k) allowed for LeafletFA jobs only (Karin 2026-10-24)
#SBATCH --output=/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/logs/%x_%j.log
set -euo pipefail
PY=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
C8=/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/c8_consensus
TR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_70_30_TMS_ONLY_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5mu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
K=$($PY -c "import numpy as np; print(int(np.load('$C8/consensus_fit.npz')['K']))")
echo "[env] host $(hostname)  job ${SLURM_JOB_ID}  K $K  start $(date)"
$PY -u -W ignore::UserWarning /gpfs/commons/home/kisaev/LeafletFA/scripts/splicevi_baseline/run_leafletfa_baseline.py \
    --train "$TR" --reference-fit "$C8/consensus_fit.npz" --K "$K" --project train="$TR" --device cuda --out "$C8/estep"
echo "[env] end $(date)"
