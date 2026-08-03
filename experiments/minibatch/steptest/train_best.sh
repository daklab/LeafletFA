#!/bin/bash
#SBATCH --job-name=k20_best
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ne1dg7-[001-011]
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/steptest/train_best.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# task 12 = K20 junc seed0; RECOMMENDED recipe B=8192, 3000 steps (vs production B=2048/1500)
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -u /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa/run_training_minibatch.py 12 /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-18 /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_splice_model_ready_70pct_2026-06-17.h5ad 3000 8192 0.05 0.1 0 2>&1
cp -r /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-18/mb_run_12 /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-18/mb_run_12_best && echo "SAVED mb_run_12_best"
