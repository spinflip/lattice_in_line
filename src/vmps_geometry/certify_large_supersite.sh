#!/bin/bash
cd "$(dirname "$0")"

# Hidden-bond constraint: by DEFAULT every supersite must contain an interaction
# edge (INTRA_PER_BLOCK=1; for q=2 the blocking is a perfect matching along
# bonds) -- absorbing bonds into supersites is the point of the method. State/
# artifacts are tagged _ipb and the certificate is conditional on it. Override:
#   INTRA_PER_BLOCK=0 ./certify_large_supersite.sh CLUSTER              # unconstrained
#   INTRA_PER_BLOCK=0 MIN_INTRA=N ./certify_large_supersite.sh CLUSTER  # >= N hidden edges instead

CLUSTER="$1"
BLOCK="${2:-2}"
MIN_INTRA="${MIN_INTRA:-0}"
INTRA_PER_BLOCK="${INTRA_PER_BLOCK:-1}"
CTAG=""
[ "$MIN_INTRA" -gt 0 ] && CTAG+="_ie${MIN_INTRA}"
[ "$INTRA_PER_BLOCK" -eq 1 ] && CTAG+="_ipb"

if [ -z "$CLUSTER" ]; then
  echo "Usage: $0 <cluster> [block]" >&2
  echo "Example: $0 kagomeYcyl288_16x12 2" >&2
  exit 1
fi

# Output root; override with DATA_DIR=/path ./certify_large_supersite.sh ...
DATA_DIR="${DATA_DIR:-$HOME/vmps_geometry_data}"
mkdir -p "$DATA_DIR"

# Derive the phase-0 translation-seed geometry from the cluster name.
#   - {lattice}{L}_{Nx}x{Ny}[x{Nz}]  -> DIAG (make_diagonal Nx Ny Nz)
#   - pyrochlore32 / pyrochlore108   -> DIAG 2x2x2 / 3x3x3
#   - pyrochlore48{a-d}/64/128       -> TILTED <key> (tilted supercell)
#   - anything else (icosa, ...)     -> no seed (LATTICE empty -> phase 0 skipped)
parse_cluster() {
  python3 - "$1" <<'PY'
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
    m = re.match(r'^([A-Za-z]+)\d+_(\d+)x(\d+)(?:x(\d+))?$', name)
    if m:
        nz = m.group(4) if m.group(4) else "1"
        print(f"DIAG {m.group(1)} {m.group(2)} {m.group(3)} {nz}")
PY
}

LATTICE=""; NX=""; NY=""; NZ=""; TILTED=""
read -r MODE_TAG REST < <(parse_cluster "$CLUSTER")
if [ "$MODE_TAG" = "DIAG" ]; then
  read -r LATTICE NX NY NZ <<< "$REST"
  echo "parsed '$CLUSTER' -> lattice=$LATTICE Nx=$NX Ny=$NY Nz=$NZ (diagonal seed)"
elif [ "$MODE_TAG" = "TILTED" ]; then
  read -r LATTICE TILTED <<< "$REST"
  echo "parsed '$CLUSTER' -> lattice=$LATTICE tilted=$TILTED (tilted-supercell seed)"
else
  echo "cluster '$CLUSTER' has no parseable geometry; phase 0 seed disabled"
fi

LATTICE="$LATTICE" \
NX="$NX" \
NY="$NY" \
NZ="$NZ" \
TILTED="$TILTED" \
SEED=1 \
MODE=ss \
BLOCK="$BLOCK" \
MIN_INTRA="$MIN_INTRA" \
INTRA_PER_BLOCK="$INTRA_PER_BLOCK" \
STATE_DIR="$DATA_DIR/bw_run_ss_${CLUSTER}_block${BLOCK}" \
TIME_HEUR=3600 \
TIME_PER_K=43200 \
WORKERS=16 \
PROCS=60 \
JOBS_PER_SIDE=4 \
SAT_TIME=0 \
POLISH_TIME=1800 \
nohup ./certify_cluster.sh "$CLUSTER" >> "$DATA_DIR/certify_supersite_${CLUSTER}_block${BLOCK}${CTAG}.log" 2>&1 &

echo "launched supersite campaign for $CLUSTER (block $BLOCK${CTAG:+, constraints$CTAG}; pid $!)"
echo "  log (appended): $DATA_DIR/certify_supersite_${CLUSTER}_block${BLOCK}${CTAG}.log"