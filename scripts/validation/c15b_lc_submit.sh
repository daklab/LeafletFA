#!/usr/bin/env bash
# c15b_lc_submit.sh <full|f30|f10> <array spec, e.g. 0 or 1-131%40> <mem>   LeafCutter array for check 15b, settings as check 10
set -euo pipefail
D=$1; ARR=$2; MEM=$3; V=/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation; L=$V/c15b/diff/lc_${D}/array_jobs.txt
sbatch --parsable -J lfa_c15b_lc_${D} -p cpu --exclude='ne1dg7-[001-011]' -c 4 --mem=$MEM -t 12:00:00 --array=$ARR --output=$V/logs/%x_%A_%a.log \
  --wrap "D=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" $L); /gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/leafcutter-ds \"\$D/counts.tsv.gz\" \"\$D/groups.txt\" --output_prefix \"\$D/ds\" --baseline_group armA --num_threads 4 --min_samples_per_intron 4 --min_samples_per_group 4"
