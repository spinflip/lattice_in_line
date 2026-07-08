#!/bin/bash
# certify_lex_small.sh — small-cluster launcher for the lexicographic J1 -> J2
# bandwidth campaign. It wraps the master driver certify_lex_cluster.sh with
# parameters tuned for small clusters, which CP-SAT can actually CERTIFY: a
# modest per-k budget and the independent SAT cross-check enabled (SAT_TIME>0).
# The J2 graph defaults to the same-named entry in cluster_edges_NNN.py.
#
# (The large-cluster counterpart is certify_large_lex.sh; for huge tori that
#  cannot be certified, use certify_lex_large.sh instead.)
#
# Prerequisite: each cluster's J1 bandwidth must be in the shared data dir
# (run ./certify_small.sh <cluster> first), or pass K1=<cap>.
#
# Usage:
#   ./certify_lex_small.sh                 # default small-cluster batch
#   ./certify_lex_small.sh CLUSTER         # one cluster (J2 = its NNN)
#   ./certify_lex_small.sh CLUSTER J2      # explicit J2 (cluster name or edge file)
# Any of the master driver's env vars (SOFTEN_MAX, TIME_PER_K, ...) may be
# overridden on the command line; the defaults below suit small clusters.
cd "$(dirname "$0")"

J1="${1:-}"
J2="${2:-}"

DEFAULT_CLUSTERS=(
  pyrochlore32 pyrochlore48a pyrochlore48b pyrochlore48c pyrochlore48d pyrochlore64
  trillium32_2x2x2 trillium48_3x2x2 trillium64_4x2x2 trillium72_3x3x2
  hyperkagome48_2x2x1 hyperkagome48_4x1x1 hyperkagome60_5x1x1
  hyperkagome72_3x2x1 hyperkagome72_6x1x1
  kagomeBtorus48_4x4 triangularBtorus64_8x8
)

if [[ -n "$J1" ]]; then
  CLUSTERS=("$J1")
else
  CLUSTERS=("${DEFAULT_CLUSTERS[@]}")
fi

# Output root; override with DATA_DIR=/path ./certify_lex_small.sh ...
DATA_DIR="${DATA_DIR:-$HOME/vmps_geometry_data}"
mkdir -p "$DATA_DIR"

# small-cluster defaults (overridable from the environment)
TIME_PER_K="${TIME_PER_K:-14400}"
SAT_TIME="${SAT_TIME:-3600}"
WORKERS="${WORKERS:-16}"
SEED="${SEED:-1}"
SYMMETRY="${SYMMETRY:-reversal}"

# show a representative plan on the terminal (same budgets for every cluster)
plan_args=("${CLUSTERS[0]}"); [ -n "$J2" ] && plan_args+=("$J2")
env STATE_DIR="$DATA_DIR/bw_run_${CLUSTERS[0]}" TIME_PER_K="$TIME_PER_K" \
  SAT_TIME="$SAT_TIME" WORKERS="$WORKERS" SEED="$SEED" SYMMETRY="$SYMMETRY" \
  PLAN_ONLY=1 ./certify_lex_cluster.sh "${plan_args[@]}"

nohup bash -c '
  DATA_DIR="$1"
  TIME_PER_K="$2"; SAT_TIME="$3"; WORKERS="$4"; SEED="$5"; SYMMETRY="$6"; J2="$7"
  shift 7
  for c in "$@"; do
    echo "=== lex $c ==="
    args=("$c"); [ -n "$J2" ] && args+=("$J2")
    STATE_DIR="$DATA_DIR/bw_run_$c" \
    TIME_PER_K="$TIME_PER_K" \
    SAT_TIME="$SAT_TIME" \
    WORKERS="$WORKERS" \
    SEED="$SEED" \
    SYMMETRY="$SYMMETRY" \
      ./certify_lex_cluster.sh "${args[@]}" >> "$DATA_DIR/certify_lex_$c.log" 2>&1
  done
' bash "$DATA_DIR" "$TIME_PER_K" "$SAT_TIME" "$WORKERS" "$SEED" "$SYMMETRY" "$J2" \
    "${CLUSTERS[@]}" >> "$DATA_DIR/certify_lex_small.log" 2>&1 &

echo "launched lex small campaign for ${#CLUSTERS[@]} cluster(s) (pid $!)"
for c in "${CLUSTERS[@]}"; do echo "  campaign log: $DATA_DIR/certify_lex_$c.log"; done
echo "  (launcher wrapper log: $DATA_DIR/certify_lex_small.log)"
