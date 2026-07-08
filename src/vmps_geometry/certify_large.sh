#!/bin/bash
cd "$(dirname "$0")"

CLUSTER="$1"

if [ -z "$CLUSTER" ]; then
  echo "Usage: $0 <cluster>" >&2
  echo "Example: $0 pyrochlore128" >&2
  exit 1
fi

DATA_DIR="$HOME/vmps_geometry_data"
mkdir -p "$DATA_DIR"

STATE_DIR="$DATA_DIR/bw_run_${CLUSTER}" \
TIME_HEUR=14400 STALL=3600 \
TIME_OPT=14400 \
TIME_PER_K=43200 \
SAT_TIME=0 \
WORKERS=16 PROCS=160 JOBS_PER_SIDE=4 \
SEED=1 \
nohup ./certify_cluster.sh "$CLUSTER" >> "$DATA_DIR/certify_${CLUSTER}.log" 2>&1 &

## extract mid-run:
#python -m vmps_geometry.bandwidth_certifier export --cluster "$CLUSTER" --state-dir "bw_run_${CLUSTER}" > "current_best_${CLUSTER}.json"
