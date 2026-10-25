#!/usr/bin/env bash
# CPU wrapper for validation scripts.  sbatch -J v_x -c 16 --mem 40G -t 6:00:00 sbatch_cpu.sh <script.py> [args]
#SBATCH --partition=cpu
#SBATCH --exclude=ne1dg7-[001-011]
#SBATCH --output=/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/logs/%x_%j.log
set -euo pipefail
echo "[env] host $(hostname) job ${SLURM_JOB_ID} cpus ${SLURM_CPUS_PER_TASK:-?} start $(date)"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -u -W ignore::UserWarning "/gpfs/commons/home/kisaev/LeafletFA/scripts/validation/$1" "${@:2}"
echo "[env] end $(date)"
