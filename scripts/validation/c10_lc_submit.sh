#!/usr/bin/env bash
# c10_lc_submit.sh <TMS|Allen> <halfA|full> <array spec, e.g. 0 or 1-99%25> <mem>   LeafCutter array, settings as check 9b
set -euo pipefail
DS=$1; DES=$2; ARR=$3; MEM=$4; V=/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation; L=$V/c10_lc/${DS}_${DES}/array_jobs.txt
sbatch --parsable -J lfa_c10_lc_${DS}_${DES} -p cpu --exclude='ne1dg7-[001-011]' -c 4 --mem=$MEM -t 12:00:00 --array=$ARR --output=$V/logs/%x_%A_%a.log \
  --wrap "D=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" $L); /gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/leafcutter-ds \"\$D/counts.tsv.gz\" \"\$D/groups.txt\" --output_prefix \"\$D/ds\" --baseline_group armR --num_threads 4 --min_samples_per_intron 4 --min_samples_per_group 4"
