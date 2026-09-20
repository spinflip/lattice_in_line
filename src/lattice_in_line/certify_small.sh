cd "$(dirname "$0")"

# DEFAULT_CLUSTERS=(
#   icosa cubocta C12 C20 C24 C26 C28 C30 C36 C40 C60
#   pyrochlore48a pyrochlore48b pyrochlore48c pyrochlore48d pyrochlore64
#   trillium32_2x2x2 trillium48_3x2x2 trillium64_4x2x2
#   kagomeBtorus48_4x4
# )

DEFAULT_CLUSTERS=(hyperkagome48_4x1x1 hyperkagome48_2x2x1 hyperkagome60_5x1x1 hyperkagome72_6x1x1)

# Accept any number of cluster names as arguments (they run sequentially in one
# background job); with no arguments, fall back to DEFAULT_CLUSTERS.
if [[ $# -gt 0 ]]; then
  CLUSTERS=("$@")
else
  CLUSTERS=("${DEFAULT_CLUSTERS[@]}")
fi

# Output root; override with DATA_DIR=/path ./certify_small.sh [clusters...]
DATA_DIR="${DATA_DIR:-$HOME/lattice_in_line_data}"
mkdir -p "$DATA_DIR"

# peak number of CPU cores the batch may use
CPUS="${CPUS:-8}"

# MODE (plain|ss|cutwidth) is inherited by the env calls below; tag the per-
# cluster log so a cutwidth run never overwrites the bandwidth run's log.
MODE="${MODE:-cutwidth}"
case "$MODE" in
  plain)    LOGTAG="_bw" ;;
  cutwidth) LOGTAG="_cw" ;;
  *)        LOGTAG="_$MODE" ;;
esac
# run-directory prefix: cutwidth -> cw_run_, everything else -> bw_run_
RUNPREFIX=bw; [[ "$MODE" == "cutwidth" ]] && RUNPREFIX=cw

# Per-mode budgets (see certify_large.sh for the rationale): cutwidth favours the
# SA heuristic and shortens each CP-SAT decision (its UNSAT side rarely closes).
if [[ "$MODE" == "cutwidth" ]]; then
  TIME_HEUR_D=7200       # 2h SA (the workhorse)
  TIME_PER_K_D=1800      # 30min per CP-SAT decision
  LADDER_TIME_D=3600     # 1h TOTAL ladder cap (SAT probe + LB tightening)
else
  TIME_HEUR_D=3600       # 1h
  TIME_PER_K_D=14400     # 4h
  LADDER_TIME_D=0        # unlimited (bandwidth ladder closes both sides)
fi

CENV=(
  TIME_HEUR=$TIME_HEUR_D STALL=300 TIME_OPT=3600 TIME_PER_K=$TIME_PER_K_D
  LADDER_TIME=$LADDER_TIME_D
  # CPUS is the peak core count (override: CPUS=16 ./certify_small.sh ...):
  # SA uses CPUS procs; the bw ladder runs 2 x 1 x CPUS/2 CP-SAT threads.
  SAT_TIME=3600 WORKERS=$(( CPUS >= 2 ? CPUS / 2 : 1 )) PROCS=$CPUS JOBS_PER_SIDE=1 SEED=1
)
# show a representative plan (same budgets for every cluster) and confirm
if ! env "${CENV[@]}" STATE_DIR="$DATA_DIR/${RUNPREFIX}_run_${CLUSTERS[0]}" PLAN_ONLY=1 CONFIRM=1 \
     ./certify_cluster.sh "${CLUSTERS[0]}"; then
  echo "batch not launched."; exit 1
fi

nohup bash -c '
  DATA_DIR="$1"; LOGTAG="$2"; RUNPREFIX="$3"; NENV="$4"; shift 4
  CENV=( "${@:1:$NENV}" ); shift "$NENV"
  for c in "$@"; do
    echo "=== $c ==="
    env "${CENV[@]}" STATE_DIR="$DATA_DIR/${RUNPREFIX}_run_$c" \
      ./certify_cluster.sh "$c" >> "$DATA_DIR/certify_${c}${LOGTAG}.log" 2>&1
  done
' bash "$DATA_DIR" "$LOGTAG" "$RUNPREFIX" "${#CENV[@]}" "${CENV[@]}" "${CLUSTERS[@]}" > /dev/null 2>&1 &

echo "launched small campaign for ${#CLUSTERS[@]} cluster(s) (pid $!, $CPUS CPUs)"
for c in "${CLUSTERS[@]}"; do echo "  campaign log: $DATA_DIR/certify_${c}${LOGTAG}.log"; done
