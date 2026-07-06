#!/usr/bin/env bash
# certify_lex_cluster.sh — lexicographic J1 -> J2 bandwidth campaign for one
# cluster (the driver behind certify_large_lex.sh, mirroring certify_cluster.sh).
#
# Given a J1 graph (e.g. nearest-neighbour), find the chain ordering that
# minimizes the J2 graph's bandwidth k2 (e.g. next-nearest-neighbour) SUBJECT TO
# d_e <= k1 on every J1 edge. The J1 cap k1 is taken from the J1 state: the
# certified k1 if available, otherwise the best-known J1 upper bound (J1 need NOT
# be certified). Uses the certifier's lex-ladder (heuristic seed + alternating
# CP-SAT k2 decision ladder), optionally cross-checks the decisive UNSAT with two
# independent SAT solvers, and exports the resulting labeling.
#
# Usage:
#   ./certify_lex_cluster.sh J1_CLUSTER [J2_SPEC]
#     J1_CLUSTER  cluster name in cluster_edges.py (the J1 / NN graph)
#     J2_SPEC     the J2 / NNN graph (optional; defaults to J1_CLUSTER, i.e. the
#                 same-named entry in cluster_edges_NNN.py). May also be another
#                 cluster name, or a path to a 0-indexed 'u v' edge file from
#                 'vmps-cluster-generator <lat> ... --neighbor-shell 2 --format edgelist'.
#
# Tunables (env vars, defaults in parentheses):
#   STATE_DIR      state dir, SHARED with the J1 campaign  (~/vmps_geometry_data/bw_run_<J1>)
#   K1             explicit J1 bandwidth cap; empty => use the J1 state's certified
#                  k1, or its best-known upper bound if not yet certified     ()
#   SOFTEN_MAX     sweep softenings s=0..SOFTEN_MAX, running the campaign at each
#                  J1 cap k1 = base_k1 + s. Higher s loosens J1 to buy a smaller
#                  k2 (the J1<->J2 trade). Each s is independent.              (5)
#   SOFTEN_MIN     start the sweep at this s (for resuming a partial sweep)    (0)
#   SOFTEN         shortcut for a SINGLE cap: sets SOFTEN_MIN=SOFTEN_MAX=SOFTEN ()
#   TIME_PER_K     seconds per ONE CP-SAT k2 decision        (14400)
#   LADDER_TIME    wall-clock cap for each softening's whole ladder; best-so-far
#                  is still exported. 0 = no cap (needs `timeout`/`gtimeout`) (0)
#   WORKERS        CP-SAT threads per decision               (16)
#   SEED           heuristic seed                            (1)
#   SYMMETRY       navigation symmetry: orbit|reversal       (reversal)
#   SAT_TIME       decisive-k2 SAT cross-check seconds; 0 = skip            (0)
#   FINAL_SYM      symmetry for the cross-check CNF          (reversal)
#   EDGES_MODULE   J1 table module                           (cluster_edges.py)
#   J2_EDGES_MODULE J2 table module for a named J2          (cluster_edges_NNN.py)
#   CERT           path to bandwidth_certifier.py            (./bandwidth_certifier.py)
#   PYTHON         interpreter                               (python3)
#
# Requires bash. If started by sh/dash/zsh, re-exec under bash transparently.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail
ORIG_PWD="$PWD"
cd "$(dirname "$0")"

J1="${1:?usage: $0 J1_CLUSTER [J2_SPEC]}"
# J2 defaults to the same-named NNN table entry (cluster_edges_NNN.py).
J2="${2:-$J1}"

# A relative J2 edge-file path is meant relative to the user's working
# directory, not this script's dir (we just cd'd away from it). Resolve it.
if [[ "$J2" != /* && -f "$ORIG_PWD/$J2" ]]; then
  J2="$ORIG_PWD/$J2"
fi

PYTHON="${PYTHON:-python3}"
CERT="${CERT:-./bandwidth_certifier.py}"
STATE_DIR="${STATE_DIR:-$HOME/vmps_geometry_data/bw_run_${J1}}"
EDGES_MODULE="${EDGES_MODULE:-cluster_edges.py}"
J2_EDGES_MODULE="${J2_EDGES_MODULE:-cluster_edges_NNN.py}"
K1="${K1:-}"
# Softening sweep: run the campaign for every J1 cap k1 = base_k1 + s, with s
# from SOFTEN_MIN to SOFTEN_MAX (default 0..5). Each s is independent (its own
# state/labeling). For a single cap, set SOFTEN=N (-> min=max=N), back-compatible.
SOFTEN_MIN="${SOFTEN_MIN:-${SOFTEN:-0}}"
SOFTEN_MAX="${SOFTEN_MAX:-${SOFTEN:-5}}"
TIME_PER_K="${TIME_PER_K:-14400}"   # budget for ONE CP-SAT k2 decision
# Overall wall-clock cap for the lex-ladder of EACH softening (0 = no cap). A
# ladder makes many decisions, each up to TIME_PER_K, so without a cap a hard
# large-torus sweep can run for days. On a cap the best-so-far is kept/exported.
LADDER_TIME="${LADDER_TIME:-0}"
WORKERS="${WORKERS:-16}"
SEED="${SEED:-1}"
SYMMETRY="${SYMMETRY:-reversal}"
SAT_TIME="${SAT_TIME:-0}"
FINAL_SYM="${FINAL_SYM:-reversal}"

mkdir -p "$STATE_DIR"

# J2 source: an existing file => --j2-file; otherwise a named entry looked up in
# the J2 table module (cluster_edges_NNN.py by default).
if [[ -f "$J2" ]]; then
  J2_ARGS=(--j2-file "$J2")
  J2NAME="$(basename "$J2")"; J2NAME="${J2NAME%.*}"
else
  J2_ARGS=(--j2-cluster "$J2" --j2-edges-module "$J2_EDGES_MODULE")
  J2NAME="$J2"
fi

# J1 plain-campaign state (written by certify_cluster.sh), read to resolve the
# base J1 cap. Each softening's lex state/labeling is named per effective k1
# inside run_cap, so the sweep's caps never share files.
J1_JSON="$STATE_DIR/${J1}.json"

C() { "$PYTHON" "$CERT" "$@" --cluster "$J1" --state-dir "$STATE_DIR" \
        --edges-module "$EDGES_MODULE"; }

log() { echo "[$(date '+%F %T')] $*"; }

# k2 window from a lex state JSON ($1); prints "LB UB" (UB=-1 if no result)
window() {
  "$PYTHON" - "$1" <<'EOF'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print("1 -1"); raise SystemExit
st = json.load(open(p))
lb = st["math_lb"] or 1
if st["unsat"]:
    lb = max(lb, 1 + max(int(k) for k in st["unsat"]))
print(lb, st["ub"] if st["ub"] is not None else -1)
EOF
}

# proof level recorded for k2=$2 in lex state $1 (math/cpsat/xsat/drat), or "none"
proof_of() {
  "$PYTHON" - "$1" "$2" <<'EOF'
import json, sys
st = json.load(open(sys.argv[1]))
print(st["unsat"].get(sys.argv[2], "none"))
EOF
}

# best-known J1 bandwidth (the upper bound "ub") and whether it is certified
# (lb==ub). Prints "UB CERTIFIED" (UB="" if no result yet, CERTIFIED=0|1).
j1_best() {
  "$PYTHON" - "$J1_JSON" <<'EOF'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print("NONE 0"); raise SystemExit
st = json.load(open(p))
ub = st.get("ub")
lb = st.get("math_lb") or 1
if st.get("unsat"):
    lb = max(lb, 1 + max(int(k) for k in st["unsat"]))
cert = 1 if (ub is not None and lb >= ub) else 0
print(ub if ub is not None else "NONE", cert)
EOF
}

log "lex campaign: J1=$J1  J2=$J2NAME  (state: $STATE_DIR)"

# Resolve the J1 bandwidth cap k1. Priority: explicit K1, else the best-known J1
# upper bound from the J1 state (works even if J1 is NOT certified). SOFTEN then
# relaxes that cap by N positions (allow J1 bandwidth up to k1+N to buy a smaller
# k2). lex-ladder enforces d_e <= k1 on every J1 edge.
if [[ -n "$K1" ]]; then
  BASE_K1="$K1"
  log "J1 cap base: k1=$BASE_K1 (explicit K1)"
else
  read -r BASE_K1 J1_CERT < <(j1_best)
  if [[ -z "$BASE_K1" || "$BASE_K1" == "NONE" ]]; then
    log "ERROR: no J1 bandwidth result in $J1_JSON."
    log "Run ./certify_cluster.sh $J1 (or ./certify_large.sh $J1) first, or set K1=<cap>."
    exit 1
  fi
  if [[ "$J1_CERT" -eq 1 ]]; then
    log "J1 cap base: k1=$BASE_K1 (certified J1 bandwidth)"
  else
    log "J1 cap base: k1=$BASE_K1 (best-known J1 bandwidth; NOT certified)"
  fi
fi
# run the full lex campaign for ONE softening s (effective J1 cap = BASE_K1 + s).
# Each cap is independent: its state/labeling/CNF are tagged k1_<eff>. Appends a
# one-line result to SUMMARY. Never aborts the sweep — failures are logged.
run_cap() {
  local s="$1"
  local eff=$((BASE_K1 + s))
  local tag="${J1}_lex_${J2NAME}_k1_${eff}"
  local lex_json="$STATE_DIR/${J1}__lex_${J2NAME}__k1_${eff}.json"
  local assign="$STATE_DIR/${tag}_labeling.txt"
  local k1args=(--k1 "$eff")
  log "================  softening s=$s  ->  J1 cap k1=$eff  ================"

  local capnote=""
  [[ -n "$LADDER_BIN" ]] && capnote=", cap ${LADDER_TIME}s"
  log "phase 1: lex-ladder (k1=$eff, time-per-k=${TIME_PER_K}s, workers=${WORKERS}, symmetry=${SYMMETRY}${capnote})"
  # timeout can't wrap the C() function, so build the certifier call as an array
  local cmd=()
  [[ -n "$LADDER_BIN" ]] && cmd+=("$LADDER_BIN" "$LADDER_TIME")
  cmd+=("$PYTHON" "$CERT" lex-ladder
        --cluster "$J1" --state-dir "$STATE_DIR" --edges-module "$EDGES_MODULE"
        "${J2_ARGS[@]}" "${k1args[@]}"
        --time-per-k "$TIME_PER_K" --workers "$WORKERS" --seed "$SEED" --symmetry "$SYMMETRY")
  local rc=0
  "${cmd[@]}" || rc=$?
  if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
    log "softening $s: ladder hit LADDER_TIME=${LADDER_TIME}s wall-clock cap; exporting best-so-far"
  elif [[ "$rc" -ne 0 ]]; then
    log "softening $s: lex-ladder failed (rc=$rc, see above); skipping this cap"
    SUMMARY+=("$(printf '  s=%-2s  k1=%-3s  k2=%-5s  %s' "$s" "$eff" "ERR" "lex-ladder-failed")")
    return 0
  fi

  local LB UB
  read -r LB UB < <(window "$lex_json")
  if [[ "$UB" -lt 0 ]]; then
    log "softening $s: no k2 result yet (re-run to resume)"
    SUMMARY+=("$(printf '  s=%-2s  k1=%-3s  k2=%-5s  %s' "$s" "$eff" "-" "no-result")")
    return 0
  fi
  local closed=0
  [[ "$LB" -ge "$UB" ]] && closed=1

  # phase 2: optional independent SAT cross-check of the decisive UNSAT
  if [[ "$closed" -eq 1 && "${SAT_TIME%.*}" -gt 0 ]]; then
    local kdec=$((UB - 1)) pl
    pl="$(proof_of "$lex_json" "$kdec")"
    if [[ "$pl" != "xsat" && "$pl" != "drat" ]]; then
      log "phase 2: SAT cross-check of decisive k2=$kdec (${SAT_TIME}s)"
      C lex-verify "${J2_ARGS[@]}" "${k1args[@]}" --k "$kdec" --time "$SAT_TIME" \
        --symmetry "$FINAL_SYM" --cnf-out "$STATE_DIR/${tag}_k2_${kdec}.cnf" \
        --proof-out "$STATE_DIR/${tag}_k2_${kdec}.drat" \
        > "$STATE_DIR/${tag}_verify_k2_${kdec}.log" 2>&1 || true
    fi
  fi

  # phase 3: export the labeling for this cap
  C lex-export "${J2_ARGS[@]}" "${k1args[@]}" > "$assign"
  if [[ "$closed" -eq 1 ]]; then
    log "softening $s: k1=$eff -> k2*=$UB (CERTIFIED given k1); labeling: $assign"
    SUMMARY+=("$(printf '  s=%-2s  k1=%-3s  k2=%-5s  %s' "$s" "$eff" "$UB" "certified")")
  else
    log "softening $s: k1=$eff -> best k2=$UB (lb $LB, NOT certified); labeling: $assign"
    SUMMARY+=("$(printf '  s=%-2s  k1=%-3s  k2=%-5s  %s' "$s" "$eff" "$UB" "open[lb=$LB]")")
  fi
  return 0
}

# ----------------------------------------------- softening sweep s=MIN..MAX
if [[ "$SOFTEN_MIN" -gt "$SOFTEN_MAX" ]]; then
  log "ERROR: SOFTEN_MIN ($SOFTEN_MIN) > SOFTEN_MAX ($SOFTEN_MAX)"
  exit 1
fi
# resolve the per-softening wall-clock cap binary, if requested
LADDER_BIN=""
if [[ "${LADDER_TIME%.*}" -gt 0 ]]; then
  if command -v timeout >/dev/null 2>&1; then LADDER_BIN=timeout
  elif command -v gtimeout >/dev/null 2>&1; then LADDER_BIN=gtimeout
  else log "warning: LADDER_TIME=${LADDER_TIME} set but no timeout/gtimeout; cap disabled"; fi
fi
log "softening sweep: s = ${SOFTEN_MIN}..${SOFTEN_MAX}  (base J1 cap k1=$BASE_K1)"
[[ -n "$LADDER_BIN" ]] && log "per-softening wall-clock cap: ${LADDER_TIME}s (via $LADDER_BIN)"
SUMMARY=()
for ((s = SOFTEN_MIN; s <= SOFTEN_MAX; s++)); do
  run_cap "$s"
done

log "================  sweep summary  (J1=$J1, J2=$J2NAME, base k1=$BASE_K1)  ================"
for line in "${SUMMARY[@]}"; do log "$line"; done
log "  (lower k2 at higher k1 is the J1<->J2 trade-off; each cap's layout is its own"
log "   ${J1}_lex_${J2NAME}_k1_<k1>_labeling.txt)"
