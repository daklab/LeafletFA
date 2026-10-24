#!/usr/bin/env bash
# GPU wrapper for the SpliceVI-baseline scripts (L40S or b6k, whichever schedules first). Resources on the command line:
#   sbatch -J lfa_x -c 8 --mem 64G -t 2:00:00 sbatch_gpu.sh run_leafletfa_baseline.py --device cuda <args>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# ne1dg7 (b6k) allowed for LeafletFA jobs only (Karin 2026-10-24); card type is recorded per run
#SBATCH --output=/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_splicevi_baseline/logs/%x_%j.log
set -euo pipefail
echo "[env] host $(hostname)  job ${SLURM_JOB_ID}  cpus ${SLURM_CPUS_PER_TASK:-?}  gpu ${CUDA_VISIBLE_DEVICES:-none}  start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
HERE=/gpfs/commons/home/kisaev/LeafletFA/scripts/splicevi_baseline
script=$1; shift
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -u -W ignore::UserWarning "$HERE/$script" "$@"
echo "[env] end $(date)"
