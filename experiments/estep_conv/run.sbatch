#!/bin/bash
#SBATCH --job-name=estep_conv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ne1dg7-[001-011]
#SBATCH --mem=150G
#SBATCH --time=8:00:00
#SBATCH --output=/gpfs/commons/home/kisaev/LeafletFA/experiments/estep_conv/conv.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa
/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python -u 07d_estep_convergence_probe.py "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-18/run_30" "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/SMOKE_04E2/resampled/RESAMPLED_10_PERCENT_test_30_70_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5mu" "/gpfs/commons/home/kisaev/LeafletFA/experiments/estep_conv/out.csv" 4096 300,800,2000 10
