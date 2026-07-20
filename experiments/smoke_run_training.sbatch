#!/bin/bash
#SBATCH --job-name=smoke_rt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/commons/home/kisaev/LeafletFA/experiments/smoke_run_training.log
set -e
WD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION
OUT=$WD/LEAFLETFA_MODELS/SMOKE
PY=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
rm -rf $OUT; mkdir -p $OUT

echo "=== 1. build a small subset of the REAL training data ==="
$PY - <<PYEOF
import anndata as ad, numpy as np
a = ad.read_h5ad("$WD/MODEL_INPUT/20260526/train_splice_model_ready_70pct_2026-06-17.h5ad")
rng = np.random.default_rng(0)
b = a[np.sort(rng.choice(a.n_obs, 4000, replace=False))].copy()
b.write("$OUT/tiny_train.h5ad")
print(f"   {b.n_obs} cells x {b.n_vars} junctions")
PYEOF

echo "=== 2. a params file that exercises the SAME code path (pruning + save) but fast ==="
$PY - <<PYEOF
import json
p = [dict(input_conc=None, junc_specific_prior=True, delta_fixed=None, K=10,
          waypoints_use=False, seed=0, arm="smoke",
          batch_size=2000, num_passes=2, num_epochs_first=15, num_epochs_later=10,
          ELBO_num_particles=2, num_samples=20, gamma=0.5, min_delta=100,
          min_delta_rel=1e-5, patience=50, lr=0.05, max_junctions=None, num_inits=1)]
json.dump(p, open("$OUT/parameter_combinations.json","w"), indent=2)
print("   1 param set, K=10, 2 passes")
PYEOF

echo "=== 3. run run_training.py END TO END (this is the step that was never done) ==="
cd /gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa
# NB the CLI is POSITIONAL: run_training.py <task_id> <model_dir> <h5ad>
WANDB_MODE=disabled $PY -u run_training.py 0 $OUT $OUT/tiny_train.h5ad 2>&1 | tail -40

echo "=== 4. DID IT ACTUALLY SAVE? ==="
ls -la $OUT/run_0/ 2>/dev/null | head
$PY - <<PYEOF
import gzip, pickle, os
p = "$OUT/run_0/leafletfa_model.pkl.gz"
if not os.path.exists(p):
    print("   *** NO MODEL PICKLE WRITTEN — the save path is still broken ***"); raise SystemExit(1)
d = {}
with gzip.open(p, "rb") as f:
    while True:
        try: d.update(pickle.load(f))
        except EOFError: break
print("   MODEL SAVED. keys:", sorted(d.keys())[:12])
for k in ("psi", "assign_post", "pi"):
    print(f"     {k}: {getattr(d.get(k), 'shape', None)}")
print("\n*** run_training.py COMPLETES AND SAVES ***")
PYEOF
