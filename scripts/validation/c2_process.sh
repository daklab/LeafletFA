#!/usr/bin/env bash
# per check-2 config: knn_check + c1_score once its 5 projections exist (idempotent)
cd /gpfs/commons/home/kisaev/LeafletFA/scripts/validation
for cfg in K5 K10 K30 K20_priorfloor1; do
  R=/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/check2/TMS_$cfg
  n=$(ls $R/factors_*.npz 2>/dev/null | wc -l)
  [ "$n" -lt 5 ] && { echo "$cfg: $n/5 projections"; continue; }
  [ -f $R/knn_check.tsv ] || OMP_NUM_THREADS=2 /gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -W ignore ../splicevi_baseline/knn_check.py --run $R --obs /gpfs/commons/home/kisaev/SpliceVI-utils/manuscript/age_analysis/audit_2026_09/screen_evals/s170/emb/inputs/TMS_TEST_obs.npz > /dev/null
  [ -f /gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/tsv/c1_TMS_${cfg}_gains.tsv ] || OMP_NUM_THREADS=2 SLURM_CPUS_PER_TASK=2 /gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -W ignore c1_score.py --ds TMS --run $R --k1 /gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/check1/TMS/K1_s0 --tag $cfg > /gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/logs/c1_score_TMS_$cfg.log 2>&1
  echo "$cfg: processed"
done
[ -f /gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/check1/../../LeafletFA_splicevi_baseline/TMS/full_K20_s0_f64/knn_check.tsv ] && OMP_NUM_THREADS=2 /gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -W ignore c2_summary.py
