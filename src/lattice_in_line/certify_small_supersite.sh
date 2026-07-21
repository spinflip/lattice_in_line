#!/bin/bash
# certify_small_supersite.sh — batch supersite (MODE=ss) certification for the
# small / easy clusters, mirroring certify_small.sh but in supersite mode.
#
# Usage:
#   ./certify_small_supersite.sh                 # all default clusters, block 2
#   ./certify_small_supersite.sh CLUSTER         # one cluster, block 2
#   ./certify_small_supersite.sh CLUSTER BLOCK   # one cluster, given block size
#
# Hidden-bond constraint: by DEFAULT every supersite must contain an interaction
# edge (INTRA_PER_BLOCK=1; for q=2 the blocking is a perfect matching along
# bonds) -- absorbing bonds into supersites is the point of the method. State/
# artifacts are tagged _ipb and the certificate is conditional on it. Override:
#   INTRA_PER_BLOCK=0 ./certify_small_supersite.sh CLUSTER              # unconstrained
#   INTRA_PER_BLOCK=0 MIN_INTRA=N ./certify_small_supersite.sh CLUSTER  # >= N hidden edges instead
#
# Thresholds: between certify_small.sh (light) and certify_large.sh (heavy).
# Supersite decision problems are harder per solve than the plain ones, so we
# give more time per k and more SA effort than the plain small batch, but stay
# below the large-cluster budgets.
#
# The translation-blocking seed (phase 0) geometry is parsed from each cluster
# name: {lattice}{L}_{Nx}x{Ny}[x{Nz}] -> diagonal; pyrochlore32/108 -> diagonal
# 2x2x2 / 3x3x3; pyrochlore48{a-d}/64/128 -> tilted supercell. Names that don't
# match (icosa, C20, ...) skip phase 0.
cd "$(dirname "$0")"

CLUSTER="${1:-}"
BLOCK="${2:-2}"
MIN_INTRA="${MIN_INTRA:-0}"
INTRA_PER_BLOCK="${INTRA_PER_BLOCK:-1}"
CTAG=""
[[ "$MIN_INTRA" -gt 0 ]] && CTAG+="_ie${MIN_INTRA}"
[[ "$INTRA_PER_BLOCK" -eq 1 ]] && CTAG+="_ipb"

DEFAULT_CLUSTERS=(
  icosa cubocta C12 C20 C24 C26 C28 C30 C36 C40 C60
  pyrochlore48a pyrochlore48b pyrochlore48c pyrochlore48d pyrochlore64
  trillium32_2x2x2 trillium48_3x2x2 trillium64_4x2x2
  kagomeBtorus48_4x4
)

if [[ -n "$CLUSTER" ]]; then
  CLUSTERS=("$CLUSTER")
else
  CLUSTERS=("${DEFAULT_CLUSTERS[@]}")
fi

# Output root; override with DATA_DIR=/path ./certify_small_supersite.sh ...
DATA_DIR="${DATA_DIR:-$HOME/vmps_geometry_data}"
mkdir -p "$DATA_DIR"

# timeouts shared by every cluster in the batch (single source, threaded below)
CENV=(
  TIME_HEUR=3600 STALL=600 TIME_OPT=3600 TIME_PER_K=28800
  SAT_TIME=0 POLISH_TIME=1800 WORKERS=16 PROCS=40 JOBS_PER_SIDE=2 SEED=1
)
# representative plan on the terminal: parse the first cluster's seed geometry
_PARSE=$(python3 -c 'import re,sys
name=sys.argv[1]
D={"pyrochlore32":(2,2,2),"pyrochlore108":(3,3,3)}
T={"pyrochlore48a":"48a","pyrochlore48b":"48b","pyrochlore48c":"48c","pyrochlore48d":"48d","pyrochlore64":"64","pyrochlore128":"128"}
if name in D: print("DIAG pyrochlore",*D[name])
elif name in T: print("TILTED pyrochlore",T[name])
else:
 m=re.match(r"^([A-Za-z]+)\d+_(\d+)x(\d+)(?:x(\d+))?$",name)
 if m: print("DIAG",m.group(1),m.group(2),m.group(3),m.group(4) or "1")' "${CLUSTERS[0]}")
_L=""; _X=""; _Y=""; _Z=""; _T=""
read -r _MT _REST <<< "$_PARSE"
if [ "$_MT" = "DIAG" ]; then read -r _L _X _Y _Z <<< "$_REST"
elif [ "$_MT" = "TILTED" ]; then read -r _L _T <<< "$_REST"; fi
if ! env "${CENV[@]}" MODE=ss BLOCK="$BLOCK" MIN_INTRA="$MIN_INTRA" INTRA_PER_BLOCK="$INTRA_PER_BLOCK" \
     LATTICE="$_L" NX="$_X" NY="$_Y" NZ="$_Z" TILTED="$_T" \
     STATE_DIR="$DATA_DIR/bw_run_ss_${CLUSTERS[0]}_block${BLOCK}" PLAN_ONLY=1 CONFIRM=1 \
     ./certify_cluster.sh "${CLUSTERS[0]}"; then
  echo "batch not launched."; exit 1
fi

nohup bash -c '
  DATA_DIR="$1"; BLOCK="$2"; MIN_INTRA="$3"; INTRA_PER_BLOCK="$4"; CTAG="$5"; NENV="$6"; shift 6
  CENV=( "${@:1:$NENV}" ); shift "$NENV"

  parse_cluster() {
    python3 - "$1" <<"PY"
import re, sys
name = sys.argv[1]
PYRO_DIAG = {"pyrochlore32": (2,2,2), "pyrochlore108": (3,3,3)}
PYRO_TILTED = {"pyrochlore48a":"48a","pyrochlore48b":"48b","pyrochlore48c":"48c",
               "pyrochlore48d":"48d","pyrochlore64":"64","pyrochlore128":"128"}
if name in PYRO_DIAG:
    nx,ny,nz = PYRO_DIAG[name]; print(f"DIAG pyrochlore {nx} {ny} {nz}")
elif name in PYRO_TILTED:
    print(f"TILTED pyrochlore {PYRO_TILTED[name]}")
else:
    m = re.match(r"^([A-Za-z]+)\d+_(\d+)x(\d+)(?:x(\d+))?$", name)
    if m:
        nz = m.group(4) if m.group(4) else "1"
        print(f"DIAG {m.group(1)} {m.group(2)} {m.group(3)} {nz}")
PY
  }

  for c in "$@"; do
    LATTICE=""; NX=""; NY=""; NZ=""; TILTED=""
    read -r MODE_TAG REST < <(parse_cluster "$c")
    if [ "$MODE_TAG" = "DIAG" ]; then
      read -r LATTICE NX NY NZ <<< "$REST"
      echo "=== $c (block '"'"'$BLOCK'"'"', seed $LATTICE ${NX}x${NY}x${NZ}) ==="
    elif [ "$MODE_TAG" = "TILTED" ]; then
      read -r LATTICE TILTED <<< "$REST"
      echo "=== $c (block '"'"'$BLOCK'"'"', seed $LATTICE tilted=$TILTED) ==="
    else
      echo "=== $c (block '"'"'$BLOCK'"'"', no phase-0 seed) ==="
    fi
    env "${CENV[@]}" \
      LATTICE="$LATTICE" NX="$NX" NY="$NY" NZ="$NZ" TILTED="$TILTED" \
      MODE=ss BLOCK="$BLOCK" MIN_INTRA="$MIN_INTRA" INTRA_PER_BLOCK="$INTRA_PER_BLOCK" \
      STATE_DIR="$DATA_DIR/bw_run_ss_${c}_block${BLOCK}" \
      ./certify_cluster.sh "$c" \
      > "$DATA_DIR/certify_ss_${c}_block${BLOCK}${CTAG}.log" 2>&1
  done
' bash "$DATA_DIR" "$BLOCK" "$MIN_INTRA" "$INTRA_PER_BLOCK" "$CTAG" "${#CENV[@]}" "${CENV[@]}" "${CLUSTERS[@]}" \
  >> "$DATA_DIR/certify_small_supersite${CTAG}.log" 2>&1 &

echo "launched supersite batch for ${#CLUSTERS[@]} cluster(s) (block $BLOCK${CTAG:+, constraints$CTAG}; pid $!)"
for c in "${CLUSTERS[@]}"; do echo "  campaign log: $DATA_DIR/certify_ss_${c}_block${BLOCK}${CTAG}.log"; done
echo "  (launcher wrapper log: $DATA_DIR/certify_small_supersite${CTAG}.log)"
