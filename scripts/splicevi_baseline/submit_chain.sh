#!/usr/bin/env bash
# Submit a LeafletFA baseline run as a chain of N dependent Slurm jobs (afterany), so a timeout never
# loses progress: each link reruns the same command, which resumes the fit from fit/checkpoint.pt and
# skips every factors_<name>.npz already written; a link with nothing left to do exits in seconds.
#   submit_chain.sh <N links> <job name> "<sbatch resource flags>" <wrapper.sh> <script.py> [args...]
# e.g. submit_chain.sh 3 lfa_TMS "-c 8 --mem 64G -t 6:00:00" sbatch_gpu.sh run_leafletfa_baseline.py --device cuda ...
set -euo pipefail
n=$1; name=$2; res=$3; shift 3
HERE=/gpfs/commons/home/kisaev/LeafletFA/scripts/splicevi_baseline
dep=""
for i in $(seq 1 "$n"); do
    jid=$(sbatch --parsable -J "${name}_c${i}" $res $dep "$HERE/$1" "${@:2}")
    echo "link $i: $jid"
    dep="--dependency=afterany:$jid"
done
