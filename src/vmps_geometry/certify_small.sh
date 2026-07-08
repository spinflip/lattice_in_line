cd "$(dirname "$0")"

CLUSTER="${1:-}"

# DEFAULT_CLUSTERS=(
#   icosa cubocta C12 C20 C24 C26 C28 C30 C36 C40 C60
#   pyrochlore48a pyrochlore48b pyrochlore48c pyrochlore48d pyrochlore64
#   trillium32_2x2x2 trillium48_3x2x2 trillium64_4x2x2
#   kagomeBtorus48_4x4
# )

DEFAULT_CLUSTERS=(hyperkagome48_4x1x1 hyperkagome48_2x2x1 hyperkagome60_5x1x1 hyperkagome72_6x1x1)

if [[ -n "$CLUSTER" ]]; then
  CLUSTERS=("$CLUSTER")
else
  CLUSTERS=("${DEFAULT_CLUSTERS[@]}")
fi

# Output root; override with DATA_DIR=/path ./certify_small.sh [clusters...]
DATA_DIR="${DATA_DIR:-$HOME/vmps_geometry_data}"
mkdir -p "$DATA_DIR"

CENV=(
  TIME_HEUR=3600 STALL=300 TIME_OPT=3600 TIME_PER_K=14400
  SAT_TIME=3600 WORKERS=16 PROCS=160 JOBS_PER_SIDE=2 SEED=1
)
# show a representative plan on the terminal (same budgets for every cluster)
env "${CENV[@]}" STATE_DIR="$DATA_DIR/bw_run_${CLUSTERS[0]}" PLAN_ONLY=1 \
  ./certify_cluster.sh "${CLUSTERS[0]}"

nohup bash -c '
  DATA_DIR="$1"; NENV="$2"; shift 2
  CENV=( "${@:1:$NENV}" ); shift "$NENV"
  for c in "$@"; do
    echo "=== $c ==="
    env "${CENV[@]}" STATE_DIR="$DATA_DIR/bw_run_$c" \
      ./certify_cluster.sh "$c" >> "$DATA_DIR/certify_$c.log" 2>&1
  done
' bash "$DATA_DIR" "${#CENV[@]}" "${CENV[@]}" "${CLUSTERS[@]}" >> "$DATA_DIR/certify_small.log" 2>&1 &

echo "launched small campaign for ${#CLUSTERS[@]} cluster(s) (pid $!)"
for c in "${CLUSTERS[@]}"; do echo "  campaign log: $DATA_DIR/certify_$c.log"; done
echo "  (launcher wrapper log: $DATA_DIR/certify_small.log)"
