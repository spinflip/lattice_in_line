#!/bin/bash
# certify_lex_large.sh — large-cluster launcher for the lexicographic J1 -> J2
# bandwidth campaign. It calls the master driver certify_lex_cluster.sh with
# more resources and a wall-clock budget sized to finish in ~12-24h, using at
# most 160 CPUs. The J2 graph defaults to the same-named entry in
# cluster_edges_NNN.py. (Small-cluster counterpart: certify_lex_small.sh.)
#
# Time budget: the driver sweeps SOFTEN_MAX+1 softenings (J1 caps base..base+
# SOFTEN_MAX). EACH softening / relaxed k1 value gets a 10h wall-clock budget
# (LADDER_TIME=36000s); it may finish early if its ladder runs out of moves.
# Overall ceiling = (SOFTEN_MAX+1) * 10h; set SOFTEN_MAX to bound how many
# relaxed values you sweep. CPU use peaks at WORKERS (one CP-SAT solve at a
# time) <= 160.
#
# Prerequisite: the cluster's J1 bandwidth must be in the shared data dir
# (run ./certify_large.sh <cluster> first), or pass K1=<cap>.
#
# Usage:
#   ./certify_lex_large.sh <J1_cluster> [J2_cluster_or_edgefile]
cd "$(dirname "$0")"

J1="$1"
J2="$2"

if [ -z "$J1" ]; then
  echo "Usage: $0 <J1_cluster> [J2_cluster_or_edgefile]" >&2
  echo "Example: $0 kagomeYcyl288_16x12" >&2
  exit 1
fi

DATA_DIR="$HOME/vmps_geometry_data"
mkdir -p "$DATA_DIR"

args=("$J1")
[ -n "$J2" ] && args+=("$J2")

STATE_DIR="$DATA_DIR/bw_run_${J1}" \
SOFTEN_MAX="${SOFTEN_MAX:-5}" \
TIME_PER_K="${TIME_PER_K:-7200}" \
LADDER_TIME="${LADDER_TIME:-36000}" \
SAT_TIME="${SAT_TIME:-0}" \
WORKERS="${WORKERS:-160}" \
SEED="${SEED:-1}" \
SYMMETRY="${SYMMETRY:-reversal}" \
nohup ./certify_lex_cluster.sh "${args[@]}" >> "$DATA_DIR/certify_lex_${J1}.log" 2>&1 &

echo "launched lex large campaign for $J1 (pid $!); 10h per relaxed k1 value, <=160 CPUs"
echo "  log (appended): $DATA_DIR/certify_lex_${J1}.log"
