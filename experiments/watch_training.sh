#!/bin/bash
# Poll the LeafletFA training array and log health. Survives an agent session ending.
#   nohup bash experiments/watch_training.sh 18850569 &
JOB=${1:-18850569}
DIR=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-13
LOG=$DIR/WATCH.log
mkdir -p "$DIR"
echo "=== watch started $(date) for job $JOB ===" >> "$LOG"

while true; do
  TS=$(date '+%Y-%m-%d %H:%M')
  PEND=$(squeue -j "$JOB" -h -t PENDING 2>/dev/null | wc -l)
  RUN=$(squeue -j "$JOB" -h -t RUNNING 2>/dev/null | wc -l)
  DONE=$(sacct -j "$JOB" -n -X --format=State 2>/dev/null | grep -c COMPLETED)
  FAIL=$(sacct -j "$JOB" -n -X --format=State 2>/dev/null | grep -cE "FAILED|TIMEOUT|OUT_OF_ME|CANCELLED")

  echo "[$TS] pending=$PEND running=$RUN completed=$DONE failed=$FAIL" >> "$LOG"

  # THE health gate: psi_contrast must climb well above ~0.05, and epochs/batch should be ~60 not ~10
  for f in "$DIR"/logs/*.out; do
    [ -e "$f" ] || continue
    t=$(basename "$f")
    pc=$(grep -h "psi_contrast (programs differ" "$f" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    ep=$(grep -h "Training completed after" "$f" 2>/dev/null | grep -oE "[0-9]+" | tail -20 | \
         awk '{s+=$1; n++} END{if(n) printf "%.0f", s/n}')
    hc=$(grep -hc "HEALTH CHECK FAILED" "$f" 2>/dev/null)
    if [ -n "$pc" ]; then
      flag=""
      awk -v v="$pc" 'BEGIN{exit !(v < 0.05)}' && flag=" *** psi_contrast LOW — possible collapse"
      [ "$hc" != "0" ] && flag="$flag *** $hc health failures"
      echo "         $t  psi_contrast=$pc  mean_epochs/batch=${ep:-?}$flag" >> "$LOG"
    fi
  done

  # all done?
  if [ "$PEND" -eq 0 ] && [ "$RUN" -eq 0 ] && [ "$((DONE+FAIL))" -gt 0 ]; then
    echo "[$TS] ALL TASKS FINISHED. completed=$DONE failed=$FAIL" >> "$LOG"
    break
  fi
  sleep 900   # 15 min
done
