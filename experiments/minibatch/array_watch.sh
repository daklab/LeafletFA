#!/bin/bash
# Overnight watchdog for the 48-config MINIBATCH array (19286917).
# - auto-requeues a task ONCE on CUDA OOM (with expandable_segments), like the chunked watcher
# - as each task completes, reads mb_run_<id>/result.json and flags any run where psi did NOT land
#   (psi_location_err > 0.06), collapsed (psi_contrast < 0.04), or hit a NaN (n_finite<total)
# - everything else is logged, not touched.

ARRAY=19286917
MD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-18
LOG=/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/array_watch.log
RETRIED=/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/.array_retried
CHECKED=/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/.array_checked
PY=/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/python
touch "$RETRIED" "$CHECKED"
say(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
say "=== minibatch array watchdog started (array=$ARRAY) ==="

while true; do
  run_n=$(squeue -j $ARRAY -h -r -t RUNNING -o "%i" 2>/dev/null | wc -l)
  pend_n=$(squeue -j $ARRAY -h -r -t PENDING -o "%i" 2>/dev/null | wc -l)

  # OOM requeue (once)
  for tid in $(sacct -j $ARRAY -n -X --format=JobID,State 2>/dev/null \
               | awk '$2=="FAILED"||$2=="OUT_OF_MEMORY"||$2=="NODE_FAIL"{print $1}' \
               | grep -oE "_[0-9]+$" | tr -d _); do
    grep -qx "$tid" "$RETRIED" && continue
    lg=$(ls -t $MD/mb_logs/mb_${ARRAY}_${tid}.out 2>/dev/null | head -1)
    if [ -n "$lg" ] && grep -qiE "out of memory|OutOfMemoryError" "$lg" 2>/dev/null; then
      say "task $tid OOM -> requeue once"
      sed "s/^#SBATCH --array=.*/#SBATCH --array=${tid}/" $MD/leaflet_minibatch_array.sh > $MD/mb_retry_${tid}.sh
      jid=$(sbatch --parsable $MD/mb_retry_${tid}.sh 2>/dev/null); say "  -> job $jid"
    else
      say "task $tid FAILED (not OOM) -- needs diagnosis. log: $lg"
      grep -aiE "Error|Traceback" "$lg" 2>/dev/null | head -3 >> "$LOG"
    fi
    echo "$tid" >> "$RETRIED"
  done

  # health check completed tasks (once each) via their result.json
  for d in $MD/mb_run_*/; do
    tid=$(basename "$d" | sed 's/mb_run_//')
    [ -f "$d/result.json" ] || continue
    grep -qx "$tid" "$CHECKED" && continue
    echo "$tid" >> "$CHECKED"
    $PY - "$d/result.json" "$tid" >> "$LOG" 2>/dev/null <<'PYEOF'
import json,sys
r=json.load(open(sys.argv[1])); t=sys.argv[2]
le=r.get("psi_location_err"); ct=r.get("psi_contrast")
nf=r.get("n_finite_steps"); tot=r.get("total_steps"); K=r.get("K")
flags=[]
if le is None or le>0.06: flags.append(f"psi NOT landed (err={le})")
if (ct or 0)<0.04:        flags.append(f"collapsed (contrast={ct})")
if nf is not None and tot is not None and nf!=tot: flags.append(f"NaN-truncated ({nf}/{tot})")
tag="OK  " if not flags else "FLAG"
import time
print(f"[{time.strftime('%m-%d %H:%M')}] task {t} K={K} {tag} "
      f"psi_mean={r.get('psi_mean'):.4f} err={le:.4f} contrast={ct:.4f} perp={r.get('phi_perplexity'):.1f}"
      + (("  !! "+"; ".join(flags)) if flags else ""))
PYEOF
  done

  n_done=$(ls -d $MD/mb_run_*/ 2>/dev/null | wc -l)
  say "array: ${run_n} running, ${pend_n} pending, ${n_done} result.json written"
  if [ "$run_n" -eq 0 ] && [ "$pend_n" -eq 0 ]; then
    say "=== array drained; final health summary ==="
    grep -aE "task [0-9]+ K=" "$LOG" | tail -60 >> "$LOG"
    ok=$(grep -ac " OK   " "$LOG"); fl=$(grep -ac " FLAG " "$LOG")
    say "=== DONE: ${ok} OK, ${fl} FLAGGED. watchdog exiting ==="
    break
  fi
  sleep 600
done
