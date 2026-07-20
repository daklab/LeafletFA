#!/bin/bash
# Overnight watchdog: atlas array + the two NMF smoke tests.
#
# Auto-fixes ONLY the mechanical failure class we have already diagnosed (CUDA OOM from GPU
# node contention -- see task 28, which OOM'd when it landed on a GPU already holding 37/44 GB).
# Anything else is logged loudly and left alone, because a code error needs diagnosis, not a
# blind requeue. Each task is retried AT MOST ONCE so a genuinely broken config cannot loop.

MD=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-18
SC=/gpfs/commons/home/kisaev/LeafletFA/experiments/nmf_smoke
LOG=/gpfs/commons/home/kisaev/LeafletFA/experiments/overnight.log
RETRIED=/gpfs/commons/home/kisaev/LeafletFA/experiments/.overnight_retried
ARRAY=19067523
touch "$RETRIED"

say() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

say "=== overnight watchdog started (array=$ARRAY) ==="

while true; do
  # ---------- atlas array ----------
  done_n=$(sacct -j $ARRAY -n -X --format=State 2>/dev/null | grep -c COMPLETED)
  run_n=$(squeue -j $ARRAY -h -r -t RUNNING -o "%i" 2>/dev/null | wc -l)
  pend_n=$(squeue -j $ARRAY -h -r -t PENDING -o "%i" 2>/dev/null | wc -l)

  # newly failed tasks -> requeue once if (and only if) the cause was OOM
  for tid in $(sacct -j $ARRAY -n -X --format=JobID,State 2>/dev/null \
               | awk '$2=="FAILED" || $2=="OUT_OF_MEMORY" || $2=="NODE_FAIL" {print $1}' \
               | grep -oE "_[0-9]+$" | tr -d _); do
    grep -qx "$tid" "$RETRIED" && continue
    lg=$(ls -t $MD/logs/leaflet_${ARRAY}_${tid}.out 2>/dev/null | head -1)
    if [ -n "$lg" ] && tr '\r' '\n' < "$lg" 2>/dev/null | grep -qiE "out of memory|OutOfMemoryError|CUDA out of memory"; then
      say "task $tid FAILED (CUDA OOM) -> requeueing once with expandable_segments"
      sed "s/^#SBATCH --array=.*/#SBATCH --array=${tid}/" $MD/leaflet_array_job.sh > $MD/retry_${tid}.sh
      grep -q PYTORCH_CUDA_ALLOC_CONF $MD/retry_${tid}.sh || \
        sed -i 's|^export PYTHONUNBUFFERED=1|export PYTHONUNBUFFERED=1\nexport PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True|' $MD/retry_${tid}.sh
      jid=$(sbatch --parsable $MD/retry_${tid}.sh 2>/dev/null)
      say "  -> resubmitted task $tid as job $jid"
      echo "$tid" >> "$RETRIED"
    else
      say "task $tid FAILED -- NOT an OOM. Left alone for diagnosis. log: $lg"
      tr '\r' '\n' < "$lg" 2>/dev/null | grep -aiE "^[A-Za-z]*Error|Traceback|assert" | head -3 >> "$LOG"
      echo "$tid" >> "$RETRIED"
    fi
  done

  # ---------- smoke tests ----------
  for pair in "19277931:04e:$SC/04e/smoke.log:80" "19277932:07c:$SC/07c/smoke.log:150"; do
    jid=${pair%%:*}; rest=${pair#*:}; tag=${rest%%:*}; rest=${rest#*:}
    slog=${rest%%:*}; mem=${rest##*:}
    st=$(sacct -j $jid -n -X --format=State 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in
      COMPLETED)
        grep -qx "done_$tag" "$RETRIED" || { say "SMOKE $tag COMPLETED"; echo "done_$tag" >> "$RETRIED"; }
        ;;
      FAILED|OUT_OF_MEMORY|TIMEOUT|NODE_FAIL)
        grep -qx "smoke_$tag" "$RETRIED" && continue
        echo "smoke_$tag" >> "$RETRIED"
        if grep -qiE "out of memory|OutOfMemoryError|Killed" "$slog" 2>/dev/null; then
          newmem=$(( mem * 2 ))
          say "SMOKE $tag $st (OOM) -> resubmitting at ${newmem}G"
          sb=$(dirname "$slog")/run.sbatch
          sed -i "s/^#SBATCH --mem=.*/#SBATCH --mem=${newmem}G/" "$sb"
          rm -f "$(dirname "$slog")/out.csv"          # 04e/07c skip if the CSV already exists
          njid=$(sbatch --parsable "$sb" 2>/dev/null)
          say "  -> resubmitted $tag as job $njid"
        else
          say "SMOKE $tag $st -- NOT an OOM, needs diagnosis. tail:"
          tail -15 "$slog" 2>/dev/null >> "$LOG"
        fi
        ;;
    esac
  done

  say "array: ${done_n} done, ${run_n} running, ${pend_n} pending | smokes: 04e=$(squeue -j 19277931 -h -o %T 2>/dev/null || echo gone) 07c=$(squeue -j 19277932 -h -o %T 2>/dev/null || echo gone)"

  # stop once the array is fully drained AND neither smoke is queued
  if [ "$run_n" -eq 0 ] && [ "$pend_n" -eq 0 ] \
     && [ -z "$(squeue -j 19277931,19277932 -h -o %T 2>/dev/null)" ] \
     && [ -z "$(squeue -u kisaev -h -n leaflet -o %T 2>/dev/null)" ]; then
    say "=== ALL WORK DRAINED -- watchdog exiting ==="
    break
  fi
  sleep 600
done
