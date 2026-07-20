#!/bin/bash
# Kill the whole array the moment task _0 fails, so we don't burn 3 GPUs repeating a broken run.
# Three atlas arrays have now died on bugs that only surface HOURS in (device mismatch on chunk 2;
# CUDA->numpy at the final save). Task 0 is the canary.
JOB=${1:-18866425}
DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-13
LOG=$DIR/WATCHDOG.log
echo "=== watchdog on ${JOB}_0 started $(date) ===" >> "$LOG"

while true; do
  ST=$(sacct -j ${JOB}_0 -n -X --format=State 2>/dev/null | head -1 | tr -d ' ')
  case "$ST" in
    FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
      echo "[$(date '+%F %H:%M')] *** TASK 0 $ST — CANCELLING ARRAY $JOB ***" >> "$LOG"
      scancel "$JOB"
      f=$DIR/logs/leaflet_${JOB}_0.out
      echo "--- error from task 0 ---" >> "$LOG"
      tr '\r' '\n' < "$f" 2>/dev/null | grep -iE "^[A-Za-z]*Error|Traceback|RuntimeError|TypeError|ValueError|out of memory" \
        | head -5 >> "$LOG"
      exit 1 ;;
    COMPLETED)
      echo "[$(date '+%F %H:%M')] task 0 COMPLETED — array is healthy, watchdog standing down" >> "$LOG"
      exit 0 ;;
  esac
  # also catch a run that trains fine but never saves (the bug that cost 10 GPU-hours)
  sleep 300
done
