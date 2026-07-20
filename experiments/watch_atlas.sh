#!/bin/bash
# Early-warning monitor for the atlas array. Surfaces problems from the FIRST task rather than
# after all 44 finish. Every signal here corresponds to a bug that actually bit us:
#   psi_contrast < 0.04     -> symmetric collapse (the model learned nothing)
#   STUCK AT INIT           -> a global never moving off its seed value (the bb_conc bug)
#   epochs == patience      -> early stopping acting as a hard epoch cap
#   CUDA OOM / device       -> the K=50 OOM and the chunk-2 device mismatch
#   0% GPU                  -> the silent CPU-fallback on bad nodes
JOB=${1:?usage: watch_atlas.sh <jobid> <model_dir>}
MD=${2:?}
OUT=$MD/WATCH_ATLAS.log
ALERT=$MD/ALERTS.log
: > "$ALERT"
echo "=== atlas watch started $(date) job=$JOB ===" >> "$OUT"

while true; do
  TS=$(date '+%m-%d %H:%M')
  RUN=$(squeue -j "$JOB" -h -t RUNNING 2>/dev/null | wc -l)
  PEND=$(squeue -j "$JOB" -h -t PENDING 2>/dev/null | wc -l)
  DONE=$(sacct -j "$JOB" -n -X --format=State 2>/dev/null | grep -c COMPLETED)
  FAIL=$(sacct -j "$JOB" -n -X --format=State 2>/dev/null | grep -cE "FAILED|TIMEOUT|OUT_OF_ME|NODE_FAIL")
  echo "[$TS] running=$RUN pending=$PEND completed=$DONE failed=$FAIL" >> "$OUT"

  for f in "$MD"/logs/leaflet_${JOB}_*.out; do
    [ -e "$f" ] || continue
    t=$(basename "$f" .out | sed 's/.*_//')
    C=$(tr '\r' '\n' < "$f" 2>/dev/null)

    pc=$(echo "$C"   | grep -oE "psi_contrast[^:]*: *[0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
    hf=$(echo "$C"   | grep -c "HEALTH CHECK FAILED")
    stuck=$(echo "$C"| grep -c "STUCK AT INIT")
    ep=$(echo "$C"   | grep -oE "Training completed after [0-9]+" | grep -oE "[0-9]+$" | tail -5 | tr '\n' ',')
    err=$(echo "$C"  | grep -oiE "out of memory|two devices|can't convert cuda|refusing to run on CPU" | head -1)
    pass=$(echo "$C" | grep -oE "PASS [0-9]+/[0-9]+" | tail -1)

    line="   task $t  $pass  psi_contrast=${pc:-?}  epochs=${ep:-?}"
    [ "$hf" -gt 0 ]    && line="$line  HEALTHFAIL=$hf"
    [ "$stuck" -gt 0 ] && line="$line  STUCK=$stuck"
    [ -n "$err" ]      && line="$line  ERR=$err"
    echo "$line" >> "$OUT"

    # ── ALERTS: things that mean STOP AND LOOK ───────────────────────────────────────────
    if [ -n "$pc" ] && awk -v v="$pc" 'BEGIN{exit !(v < 0.04)}'; then
      echo "[$TS] task $t COLLAPSE: psi_contrast=$pc < 0.04" >> "$ALERT"; fi
    [ "$stuck" -gt 0 ] && echo "[$TS] task $t GLOBAL STUCK AT INIT ($stuck)" >> "$ALERT"
    [ -n "$err" ]      && echo "[$TS] task $t ERROR: $err" >> "$ALERT"
    # epochs equal to patience on repeat = early stopping acting as a hard cap
    echo "$ep" | grep -qE "(^|,)50,50,50" && echo "[$TS] task $t EPOCH CAP: epochs stuck at 50" >> "$ALERT"
  done

  # fail fast: if the FIRST task dies, the whole grid is probably doomed
  s0=$(sacct -j "${JOB}_0" -n -X --format=State 2>/dev/null | head -1 | tr -d ' ')
  case "$s0" in FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
      echo "[$TS] *** TASK 0 $s0 -- cancelling array to avoid burning 44 runs ***" >> "$ALERT"
      scancel "$JOB"; exit 1 ;;
  esac

  [ "$PEND" -eq 0 ] && [ "$RUN" -eq 0 ] && [ "$((DONE+FAIL))" -gt 0 ] && {
      echo "[$TS] ALL DONE completed=$DONE failed=$FAIL" >> "$OUT"; exit 0; }
  sleep 600
done
