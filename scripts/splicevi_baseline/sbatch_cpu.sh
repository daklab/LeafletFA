#!/usr/bin/env bash
# CPU wrapper for the SpliceVI-baseline scripts. Resources are set on the sbatch command line, e.g.
#   sbatch -J lfa_x -c 16 --mem 40G -t 4:00:00 sbatch_cpu.sh run_leafletfa_baseline.py <args>
#SBATCH --partition=cpu
#SBATCH --exclude=ne1dg7-[001-011]
#SBATCH --output=/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_splicevi_baseline/logs/%x_%j.log
set -euo pipefail
echo "[env] host $(hostname)  job ${SLURM_JOB_ID}  cpus ${SLURM_CPUS_PER_TASK:-?}  start $(date)"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
HERE=/gpfs/commons/home/kisaev/LeafletFA/scripts/splicevi_baseline
script=$1; shift
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -u -W ignore::UserWarning "$HERE/$script" "$@"
echo "[env] end $(date)"
