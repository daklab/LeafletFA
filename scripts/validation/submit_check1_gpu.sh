#!/usr/bin/env bash
# Check 1 GPU work: E-step on each event-split half against the frozen seed-0 fit, and the K = 1 null fit.
set -euo pipefail
BL=/gpfs/commons/home/kisaev/LeafletFA/scripts/splicevi_baseline
V=/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation
B=/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_splicevi_baseline
D=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526
for DS in TMS Allen; do
  C=$([ $DS = Allen ] && echo AB || echo TMS); MEM=$([ $DS = Allen ] && echo 96G || echo 64G)
  TR=$D/train_70_30_${C}_ONLY_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5mu
  C1=$V/check1/$DS
  mkdir -p $C1/K20_s0_split; ln -sf $B/$DS/full_K20_s0_f64/factors_real.npz $C1/K20_s0_split/factors_real.npz
  cd $BL
  ./submit_chain.sh 2 v_c1split_$DS "-c 8 --mem $MEM -t 2:00:00 -o $V/logs/%x_%j.log" sbatch_gpu.sh run_leafletfa_baseline.py \
     --train $TR --reference-fit $B/$DS/full_K20_s0_f64/fit/fit.npz --project A=$C1/test_A.npz --project B=$C1/test_B.npz \
     --npz-cells $C1/test_cells.tsv --npz-var $C1/var.tsv --K 20 --seed 0 --phi-dtype float64 --device cuda --out $C1/K20_s0_split
  ./submit_chain.sh 2 v_c1K1_$DS "-c 8 --mem $MEM -t 3:00:00 -o $V/logs/%x_%j.log" sbatch_gpu.sh run_leafletfa_baseline.py \
     --train $TR --K 1 --seed 0 --phi-dtype float64 --device cuda --out $C1/K1_s0
done
