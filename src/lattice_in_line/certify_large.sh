#!/bin/bash
cd "$(dirname "$0")"

CLUSTER="$1"

if [ -z "$CLUSTER" ]; then
  echo "Usage: $0 <cluster>" >&2
  echo "Example: $0 pyrochlore128" >&2
  exit 1
fi

# Output root; override with DATA_DIR=/path ./certify_large.sh <cluster>
DATA_DIR="${DATA_DIR:-$HOME/lattice_in_line_data}"
mkdir -p "$DATA_DIR"

# MODE (plain|ss|cutwidth) is inherited by env below; tag the launcher log so a
# cutwidth run never overwrites the bandwidth run's log. e.g.
#   MODE=cutwidth ./certify_large.sh C60  ->  certify_C60_cutwidth.log
MODE="${MODE:-plain}"
LOGTAG=""; [[ "$MODE" != "plain" ]] && LOGTAG="_$MODE"
# run-directory prefix: cutwidth -> cw_run_, everything else -> bw_run_ (so a
# cutwidth campaign gets its own state dir and never mixes with the bandwidth one)
RUNPREFIX=bw; [[ "$MODE" == "cutwidth" ]] && RUNPREFIX=cw

# Per-mode time budgets. Bandwidth spends most of its budget on the CP-SAT
# decision ladder (both sides can close). Cutwidth has a WEAK lower bound, so the
# UNSAT side can't close for large graphs -- the useful result is the SA layout.
# So for cutwidth we pour time into the SA heuristic and shorten each CP-SAT
# decision (enough for SAT-side improvements, not 12h of hopeless UNSAT).
if [[ "$MODE" == "cutwidth" ]]; then
  TIME_HEUR_D=43200      # 12h SA (the workhorse; cutwidth rarely certifies at n>~30)
  TIME_PER_K_D=3600      #  1h per CP-SAT decision
  LADDER_TIME_D=7200     #  2h TOTAL ladder cap (SAT probe + LB tightening)
else
  TIME_HEUR_D=14400      #  4h
  TIME_PER_K_D=43200     # 12h
  LADDER_TIME_D=0        # unlimited (bandwidth ladder closes both sides)
fi

CENV=(
  STATE_DIR="$DATA_DIR/${RUNPREFIX}_run_${CLUSTER}"
  TIME_HEUR=$TIME_HEUR_D STALL=3600
  TIME_OPT=14400
  TIME_PER_K=$TIME_PER_K_D
  LADDER_TIME=$LADDER_TIME_D
  SAT_TIME=0
  # 32 CPUs peak: SA phase 32 procs; bw ladder 2 sides x 1 job x 16 threads = 32
  WORKERS=16 PROCS=32 JOBS_PER_SIDE=1
  SEED=1
)
# show the plan (phases + budgets) and confirm, then launch for real
if ! env "${CENV[@]}" PLAN_ONLY=1 CONFIRM=1 ./certify_cluster.sh "$CLUSTER"; then
  echo "campaign not launched."; exit 1
fi
nohup env "${CENV[@]}" ./certify_cluster.sh "$CLUSTER" >> "$DATA_DIR/certify_${CLUSTER}${LOGTAG}.log" 2>&1 &

echo "launched large campaign for $CLUSTER (pid $!)"
echo "  campaign log: $DATA_DIR/certify_${CLUSTER}${LOGTAG}.log"

## extract mid-run:
#python -m lattice_in_line.bandwidth_certifier export --cluster "$CLUSTER" --state-dir "bw_run_${CLUSTER}" > "current_best_${CLUSTER}.json"
