#!/usr/bin/env bash
# certify_cluster.sh — from-scratch, reproducible bandwidth certification
# campaign for one cluster from cluster_edges.py.
#
# Usage:
#   ./certify_cluster.sh CLUSTER_NAME
#
# Tunables (env vars, defaults in parentheses):
#   DATA_DIR      output root for the default STATE_DIR  (~/lattice_in_line_data)
#   STATE_DIR     state directory              ($DATA_DIR/bw_run_<cluster>)
#   SEED          heuristic master seed                (1)
#   TIME_HEUR     heuristic phase seconds              (3600)
#   TIME_OPT      CP-SAT optimize phase seconds        (3600)
#   TIME_PER_K    per decision-problem seconds         (14400)
#   LADDER_TIME   (cutwidth mode) overall wall-clock cap on the phase-4 CP-SAT
#                 decision ladder; 0 = unlimited. Bounds runaway ladders on dense
#                 graphs with a wide cutwidth window.                     (14400)
#   WORKERS       CP-SAT threads per decide job        (16)
#   PROCS         heuristic processes                  (40)
#   JOBS_PER_SIDE parallel decide jobs per window end  (2)
#   STALL         stop heuristic after this many secs without
#                 improvement; 0 = auto = max(60, TIME_HEUR/10)  (0)
#   SYMMETRY      navigation symmetry: orbit|reversal  (orbit)
#   ORBITS        use pynauty orbit symmetry: 1|0|auto. auto probes pynauty and
#                 falls back to reversal if it is missing or CPU-incompatible
#                 (a broken pynauty SIGILLs); 0 forces reversal, 1 forces orbit (auto)
#   FINAL_SYM     symmetry in the final CNF            (reversal)
#   MAX_ROUNDS    navigation rounds before giving up   (50)
#   SAT_TIME      cap (secs) for the final SAT cross-check (3600);
#                 SAT_TIME=0 skips the cross-check entirely
#   POLISH_TIME   phase 6: minimize total interaction range at the certified
#                 bandwidth (secs); POLISH_TIME=0 skips polishing     (3600)
#   POLISH_ONLY   1 = (cutwidth mode) skip cw-run/cross-check and only re-polish
#                 the existing layout in STATE_DIR. Resumable: each improving
#                 layout is saved as it is found, so it continues a prior polish. (0)
#   CERT          path to bandwidth_certifier.py       (./bandwidth_certifier.py)
#   PYTHON        python interpreter                   (python3)
#   PLAN_ONLY     1 = print the campaign plan and exit without running  (0)
#   CONFIRM       1 = after the plan, ask for keyboard y/N confirmation
#                 (interactive terminals only; the launchers set this)   (0)
#   YES           1 = skip that confirmation prompt (auto-accept)        (0)
#   MODE          campaign type: plain | ss | cutwidth  (plain)
#                 cutwidth minimizes cut_max (max Hamiltonian edges crossing any
#                 MPS cut = the MPO bond dimension) instead of bandwidth. Same
#                 certified machinery (SA UB + CP-SAT ladder + SAT cross-check);
#                 state/artifacts tagged _cw so they never mix with bandwidth.
#                 Note: cutwidth lower bounds are weak, so the window only closes
#                 for small clusters (~n<=25-30); larger ones give a heuristic UB.
#                 Setting BLOCK explicitly with MODE=cutwidth minimizes the
#                 BLOCKED cutwidth (supersites of BLOCK vertices = the bond
#                 dimension of the blocked MPO); MIN_INTRA/INTRA_PER_BLOCK apply
#                 and are synergistic (a hidden bond crosses no cut at all).
#                 State/artifacts then tagged _cw<BLOCK><constraint tags>.
#   BLOCK         supersite size for MODE=ss           (2)
#   MIN_INTRA     (ss mode) require >= N edges hidden inside supersites.
#                 Separate state/artifacts tagged _ie<N>; the certificate is
#                 conditional on the constraint.                          (0)
#   INTRA_PER_BLOCK (ss mode) require EVERY supersite to contain an interaction
#                 edge (q=2: the blocking is a perfect matching along bonds).
#                 ON by default -- the point of supersites is to hide bonds.
#                 State/artifacts tagged _ipb; set =0 for unconstrained.   (1)
#   LATTICE       (ss mode) generator lattice name; if set, an automatic
#                 translation-blocking seed (ss-seed) runs FIRST to provide a
#                 structured early upper bound. Also set NX/NY/NZ, or
#                 SUPERCELL, or TILTED to match the cluster.
#   GENERATOR     path to cluster_generator.py         (./cluster_generator.py)
#   SEED_HEUR     ss-seed per-blocking heuristic secs  (60)
#   SEED_OPT      ss-seed per-blocking CP-SAT secs     (120)
#
# Example for a big machine:
#   TIME_HEUR=14400 TIME_PER_K=86400 WORKERS=16 PROCS=40 JOBS_PER_SIDE=4 \
#     ./certify_cluster.sh pyrochlore128
# This script requires bash (arrays, [[ ]], process substitution). If it was
# started by sh/dash/zsh, re-execute itself under bash transparently.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail
# Unbuffer Python stdout: under nohup the log is a file (not a TTY), so Python
# block-buffers stdout and phase/decision progress can sit unflushed for hours,
# making a long campaign look stuck. Force line-buffered, live output.
export PYTHONUNBUFFERED=1

CLUSTER="${1:?usage: $0 CLUSTER_NAME}"
PYTHON="${PYTHON:-python3}"
CERT="${CERT:-./bandwidth_certifier.py}"
DATA_DIR="${DATA_DIR:-$HOME/lattice_in_line_data}"
STATE_DIR="${STATE_DIR:-$DATA_DIR/bw_run_${CLUSTER}}"
SEED="${SEED:-1}"
TIME_HEUR="${TIME_HEUR:-3600}"
TIME_OPT="${TIME_OPT:-3600}"
TIME_PER_K="${TIME_PER_K:-14400}"
# (cutwidth mode) overall wall-clock cap on the CP-SAT decision ladder, so a wide
# window on a dense graph cannot grind for tens of hours; 0 = unlimited.
LADDER_TIME="${LADDER_TIME:-14400}"
WORKERS="${WORKERS:-16}"
PROCS="${PROCS:-40}"
JOBS_PER_SIDE="${JOBS_PER_SIDE:-2}"
STALL="${STALL:-0}"
SYMMETRY="${SYMMETRY:-orbit}"
FINAL_SYM="${FINAL_SYM:-reversal}"
MAX_ROUNDS="${MAX_ROUNDS:-50}"
MODE="${MODE:-plain}"
# capture BLOCK before it is defaulted for ss mode: MODE=cutwidth is PLAIN
# unless the caller explicitly sets BLOCK (then it minimizes the BLOCKED
# cutwidth over supersites of that size, with the hidden-bond knobs applying)
CW_BLOCK="${BLOCK:-1}"
BLOCK="${BLOCK:-2}"
MIN_INTRA="${MIN_INTRA:-0}"
INTRA_PER_BLOCK="${INTRA_PER_BLOCK:-1}"
# hidden-bond constraints: flag set for the ss-* calls, and a state/artifact
# tag matching the certifier's ss_state naming (_ie<N> before _ipb)
SS_FLAGS=()
SS_TAG=""
if [[ "$MIN_INTRA" -gt 0 ]]; then
  SS_FLAGS+=(--min-intra-edges "$MIN_INTRA"); SS_TAG+="_ie${MIN_INTRA}"
fi
if [[ "$INTRA_PER_BLOCK" -eq 1 ]]; then
  SS_FLAGS+=(--intra-per-block); SS_TAG+="_ipb"
fi
LATTICE="${LATTICE:-}"
NX="${NX:-}"; NY="${NY:-}"; NZ="${NZ:-}"
SUPERCELL="${SUPERCELL:-}"; TILTED="${TILTED:-}"
GENERATOR="${GENERATOR:-./cluster_generator.py}"
SEED_HEUR="${SEED_HEUR:-60}"
SEED_OPT="${SEED_OPT:-120}"
SAT_TIME="${SAT_TIME:-${KISSAT_TIME:-3600}}"
POLISH_TIME="${POLISH_TIME:-3600}"

C() { "$PYTHON" "$CERT" "$@" --cluster "$CLUSTER" --state-dir "$STATE_DIR"; }

# Orbit symmetry (a factor-n reduction for vertex-transitive graphs) needs a
# WORKING pynauty. A pynauty built for another CPU SIGILLs ("Illegal
# instruction"), which would crash phase 1 (info --orbits). Probe it in a
# subprocess so a broken/missing pynauty degrades to reversal symmetry instead
# of taking down the whole campaign. Override with ORBITS=1 (force) or ORBITS=0
# (skip); ORBITS=auto (default) tests it.
ORBITS="${ORBITS:-auto}"
if [[ "$ORBITS" == "auto" ]]; then
  if "$PYTHON" -c 'import pynauty; pynauty.autgrp(pynauty.Graph(3))' >/dev/null 2>&1; then
    ORBITS=1
  else
    ORBITS=0
  fi
fi
if [[ "$ORBITS" -ne 1 && "$SYMMETRY" == "orbit" ]]; then
  echo "[certify] pynauty unavailable or broken; using reversal symmetry instead of orbit" >&2
  SYMMETRY="reversal"
fi

if [[ "$MODE" == "ss" ]]; then
  STATE_JSON="$STATE_DIR/${CLUSTER}__ss${BLOCK}${SS_TAG}.json"
elif [[ "$MODE" == "cutwidth" ]]; then
  # blocked cutwidth (BLOCK set explicitly) gets its own tagged state/artifacts
  CWTAG=""
  CWFLAGS=()
  if [[ "$CW_BLOCK" -gt 1 ]]; then
    CWTAG="${CW_BLOCK}${SS_TAG}"
    CWFLAGS=(--block "$CW_BLOCK" ${SS_FLAGS[@]+"${SS_FLAGS[@]}"})
  fi
  STATE_JSON="$STATE_DIR/${CLUSTER}__cw${CWTAG}.json"
else
  STATE_JSON="$STATE_DIR/$CLUSTER.json"
fi

# read the certified window from the state JSON; prints "LB UB" (UB=-1 if none)
window() {
  "$PYTHON" - "$STATE_JSON" <<'EOF'
import json, sys
st = json.load(open(sys.argv[1]))
lb = st["math_lb"] or 1
if st["unsat"]:
    lb = max(lb, 1 + max(int(k) for k in st["unsat"]))
print(lb, st["ub"] if st["ub"] is not None else -1)
EOF
}

log() { echo "[$(date '+%F %T')] $*"; }

# proof level recorded for k (math/cpsat/xsat/drat), or "none"
proof_of() {
  "$PYTHON" - "$STATE_JSON" "$1" <<'PYEOF'
import json, sys
st = json.load(open(sys.argv[1]))
print(st["unsat"].get(sys.argv[2], "none"))
PYEOF
}

# --------- campaign plan: the phases that will run, their time budgets (in
# seconds and hours) and the CPUs each uses. PLAN_ONLY=1 prints it and exits
# without doing any work; CONFIRM=1 additionally asks for keyboard confirmation
# (the launchers use both to show the plan and confirm before backgrounding).
_dur() { awk -v s="${1%.*}" 'BEGIN{printf "%ds (%.2fh)", s, s/3600}'; }
_budget() { [[ "${1%.*}" -gt 0 ]] && _dur "$1" || echo "skipped"; }
print_plan() {
  local ladder_cpus=$(( 2 * JOBS_PER_SIDE * WORKERS ))
  log "================  campaign plan  ================"
  if [[ "$MODE" == "ss" ]]; then
    log "cluster $CLUSTER : supersite bandwidth, block $BLOCK, symmetry $SYMMETRY"
    log "  hidden-bond constraints : ${SS_TAG:-none (unconstrained)}"
    log "  state dir               : $STATE_DIR"
    if [[ -n "$LATTICE" && "$BLOCK" -eq 2 ]]; then
      log "  phase 0  translation-blocking seed  : $(_dur "$SEED_HEUR") heuristic + $(_dur "$SEED_OPT") CP-SAT per direction   CPUs: ${PROCS} SA / ${WORKERS} CP-SAT"
    else
      log "  phase 0  translation-blocking seed  : skipped (needs LATTICE and block 2)"
    fi
    log "  phase 1  heuristic SA upper bound   : $(_dur "$TIME_HEUR")  (stall ${STALL}s)   CPUs: ${PROCS}  (parallel SA chains)"
    log "  phase 1  CP-SAT decision ladder     : $(_dur "$TIME_PER_K") per k-decision   CPUs: ${WORKERS}  (solver threads)"
    log "  phase 5  SAT cross-check (if closed): $(_budget "$SAT_TIME")   CPUs: 2  (two independent solvers)"
    log "  phase 6  range polish at final bw   : $(_budget "$POLISH_TIME")   CPUs: ${WORKERS}"
  elif [[ "$MODE" == "cutwidth" ]]; then
    log "cluster $CLUSTER : cutwidth (MPO bond dimension), symmetry $SYMMETRY"
    if [[ "$CW_BLOCK" -gt 1 ]]; then
      log "  blocked   : supersites of $CW_BLOCK, hidden-bond constraints ${SS_TAG:-none}"
    fi
    log "  state dir : $STATE_DIR   (artifacts tagged _cw${CWTAG})"
    log "  note      : cutwidth lower bounds are weak; window closes only for"
    log "              small clusters (~n<=25-30), else a heuristic upper bound"
    log "  phase 1  heuristic SA upper bound   : $(_dur "$TIME_HEUR")  (stall ${STALL}s)   CPUs: ${PROCS}  (parallel SA chains)"
    log "  phase 4  CP-SAT decision ladder     : $(_dur "$TIME_PER_K") per k-decision, total cap $(_budget "$LADDER_TIME")   CPUs: ${WORKERS}  (solver threads)"
    log "  phase 5  SAT cross-check (if closed): $(_budget "$SAT_TIME")   CPUs: 2  (two independent solvers)"
    log "  phase 6  range polish at final cut  : $(_budget "$POLISH_TIME")   CPUs: ${WORKERS}"
  else
    log "cluster $CLUSTER : plain single-site bandwidth, symmetry $SYMMETRY"
    log "  state dir : $STATE_DIR"
    log "  phase 1  structural bounds (info)   : instant   CPUs: 1"
    log "  phase 2  heuristic upper bound (SA) : $(_dur "$TIME_HEUR")  (stall ${STALL}s)   CPUs: ${PROCS}  (parallel SA chains)"
    log "  phase 3  CP-SAT optimize pass       : $(_dur "$TIME_OPT")   CPUs: ${WORKERS}  (solver threads)"
    log "  phase 4  decision ladder            : $(_dur "$TIME_PER_K") per k-decision, <= ${MAX_ROUNDS} rounds   CPUs: up to ${ladder_cpus}  (2 sides x ${JOBS_PER_SIDE} jobs x ${WORKERS} threads)"
    log "  phase 5  SAT cross-check (if closed): $(_budget "$SAT_TIME")   CPUs: 2  (two independent solvers)"
    log "  phase 6  range polish at final bw   : $(_budget "$POLISH_TIME")   CPUs: ${WORKERS}"
  fi
  log "================================================"
}
print_plan
# optional keyboard confirmation (interactive terminals only); YES=1 skips it
if [[ "${CONFIRM:-0}" == 1 && "${YES:-0}" != 1 && -t 0 ]]; then
  printf '%s' "Proceed with this campaign? [y/N] " > /dev/tty
  read -r _ans < /dev/tty || _ans=""
  case "$_ans" in
    [yY]|[yY][eE][sS]) log "confirmed; launching." ;;
    *) log "aborted by user; nothing launched."; exit 3 ;;
  esac
fi
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then exit 0; fi

# ================================================================ ss mode
if [[ "$MODE" == "ss" ]]; then
  log "supersite campaign: block size $BLOCK"
  [[ -n "$SS_TAG" ]] && log "hidden-bond constraints active (state tag ${SS_TAG}): certificates are conditional on them"
  # phase 0: automatic translation-blocking seed (only with lattice params,
  # only for block size 2 — the translation involution is a q=2 pairing).
  if [[ -n "$LATTICE" && "$BLOCK" -eq 2 ]]; then
    log "phase 0: translation-blocking seed (axis-aligned + tilted directions)"
    SEED_ARGS=(--generator "$GENERATOR" --lattice "$LATTICE"
               --heur-time "$SEED_HEUR" --opt-time "$SEED_OPT"
               --procs "$PROCS" --workers "$WORKERS")
    if [[ -n "$SUPERCELL" ]]; then
      SEED_ARGS+=(--supercell="$SUPERCELL")
    elif [[ -n "$TILTED" ]]; then
      SEED_ARGS+=(--tilted "$TILTED")
    else
      SEED_ARGS+=(--Nx "$NX" --Ny "$NY" --Nz "$NZ")
    fi
    C ss-seed "${SEED_ARGS[@]}" ${SS_FLAGS[@]+"${SS_FLAGS[@]}"} \
      || log "translation seed failed (non-fatal)"
  fi
  C ss-run --block "$BLOCK" --heur-time "$TIME_HEUR" \
    --time-per-k "$TIME_PER_K" --workers "$WORKERS" --procs "$PROCS" \
    --stall "$STALL" --seed "$SEED" --symmetry "$SYMMETRY" \
    ${SS_FLAGS[@]+"${SS_FLAGS[@]}"}
  read -r LB UB < <(window)
  log "supersite window: [$LB, $UB]"
  if [[ "$UB" -lt 0 ]]; then
    log "no upper bound found yet; nothing to export. Re-run to resume."
    exit 0
  fi
  CLOSED=0
  if [[ "$LB" -ge "$UB" ]]; then
    CLOSED=1
    log "window closed: supersite bandwidth k* = $UB (certified)"
  else
    log "window NOT closed: $LB <= k* <= $UB. Proceeding with the best known"
    log "layout (bandwidth $UB); re-run to resume or farm ss-decide jobs to tighten."
  fi
  # phase 5: cross-check the decisive UNSAT only when the window is closed
  if [[ "$CLOSED" -eq 1 ]]; then
    KDEC=$((UB - 1))
    PROOF_LEVEL="$(proof_of "$KDEC")"
    if [[ "$PROOF_LEVEL" == "xsat" || "$PROOF_LEVEL" == "drat" ]]; then
      log "decisive k=$KDEC already verified at level '$PROOF_LEVEL'; skipping"
    elif [[ "${SAT_TIME%.*}" -le 0 ]]; then
      log "SAT cross-check skipped (SAT_TIME=0); certification stands at cpsat"
    else
      CNF="$STATE_DIR/${CLUSTER}_ss${BLOCK}${SS_TAG}_k${KDEC}.cnf"
      DRAT="$STATE_DIR/${CLUSTER}_ss${BLOCK}${SS_TAG}_k${KDEC}.drat"
      C ss-verify --block "$BLOCK" --k "$KDEC" --time "$SAT_TIME" \
        --symmetry "$FINAL_SYM" --cnf-out "$CNF" --proof-out "$DRAT" \
        ${SS_FLAGS[@]+"${SS_FLAGS[@]}"} \
        > "$STATE_DIR/ss_verify${SS_TAG}_k${KDEC}.log" 2>&1 || true
      if grep -q 'recorded (xsat)' "$STATE_DIR/ss_verify${SS_TAG}_k${KDEC}.log"; then
        log "cross-check passed (xsat); DRAT archived at $DRAT"
      else
        log "cross-check inconclusive; certification stands at cpsat level"
      fi
    fi
  fi
  # phase 6: polish the best known layout (ALWAYS, closed or not) at its
  # current bandwidth UB. Holding d <= UB can only preserve or improve it.
  if [[ "${POLISH_TIME%.*}" -gt 0 ]]; then
    log "phase 6: polishing total interaction range at bandwidth $UB ($(_dur "$POLISH_TIME"))"
    C polish --block "$BLOCK" --target "$UB" --time "$POLISH_TIME" \
      --workers "$WORKERS" ${SS_FLAGS[@]+"${SS_FLAGS[@]}"}
  fi
  C ss-export --block "$BLOCK" ${SS_FLAGS[@]+"${SS_FLAGS[@]}"} \
    > "$STATE_DIR/${CLUSTER}_ss${BLOCK}${SS_TAG}_assignment.txt"
  if [[ "$CLOSED" -eq 1 ]]; then
    log "result: supersite bandwidth k* = $UB (CERTIFIED); assignment in"
  else
    log "result: best-known supersite bandwidth $UB (lower bound $LB, NOT certified);"
    log "  assignment in"
  fi
  log "  $STATE_DIR/${CLUSTER}_ss${BLOCK}${SS_TAG}_assignment.txt"
  exit 0
fi

# ========================================================== cutwidth mode
if [[ "$MODE" == "cutwidth" ]]; then
  log "cutwidth campaign: minimizing cut_max (MPO bond dimension)"
  if [[ "$CW_BLOCK" -gt 1 ]]; then
    log "blocked: supersites of $CW_BLOCK (hidden-bond tag ${SS_TAG:-none});"
    log "  the objective is the bond dimension of the BLOCKED MPO"
  fi
  # POLISH_ONLY=1 skips cw-run and the cross-check and only re-polishes an
  # existing campaign's layout (send previous cw_run_* results through phase 6).
  if [[ "${POLISH_ONLY:-0}" == 1 ]]; then
    log "POLISH_ONLY=1: skipping cw-run/cross-check; polishing the existing layout"
  else
    # cw-run does the whole thing: math LB, SA upper bound, then the alternating
    # CP-SAT decision ladder. STALL feeds the heuristic; PROCS the SA pool.
    C cw-run --heur-time "$TIME_HEUR" --time-per-k "$TIME_PER_K" \
      --ladder-time "$LADDER_TIME" \
      --workers "$WORKERS" --procs "$PROCS" --stall "$STALL" --seed "$SEED" \
      --symmetry "$SYMMETRY" ${CWFLAGS[@]+"${CWFLAGS[@]}"}
  fi
  read -r LB UB < <(window)
  log "cutwidth window: [$LB, $UB]"
  if [[ "$UB" -lt 0 ]]; then
    log "no upper bound found yet; nothing to export. Re-run to resume."
    exit 0
  fi
  CLOSED=0
  if [[ "$LB" -ge "$UB" ]]; then
    CLOSED=1
    log "window closed: cutwidth c* = $UB (certified)"
  else
    log "window NOT closed: $LB <= c* <= $UB (expected for larger clusters --"
    log "cutwidth lower bounds are weak). Proceeding with the best known layout."
  fi
  # phase 5: cross-check the decisive UNSAT only when the window is closed
  if [[ "$CLOSED" -eq 1 && "${POLISH_ONLY:-0}" != 1 ]]; then
    KDEC=$((UB - 1))
    PROOF_LEVEL="$(proof_of "$KDEC")"
    if [[ "$PROOF_LEVEL" == "xsat" || "$PROOF_LEVEL" == "drat" ]]; then
      log "decisive k=$KDEC already verified at level '$PROOF_LEVEL'; skipping"
    elif [[ "${SAT_TIME%.*}" -le 0 ]]; then
      log "SAT cross-check skipped (SAT_TIME=0); certification stands at cpsat"
    else
      CNF="$STATE_DIR/${CLUSTER}_cw${CWTAG}_k${KDEC}.cnf"
      DRAT="$STATE_DIR/${CLUSTER}_cw${CWTAG}_k${KDEC}.drat"
      C cw-verify --k "$KDEC" --time "$SAT_TIME" --symmetry "$FINAL_SYM" \
        --cnf-out "$CNF" --proof-out "$DRAT" ${CWFLAGS[@]+"${CWFLAGS[@]}"} \
        > "$STATE_DIR/cw_verify${CWTAG}_k${KDEC}.log" 2>&1 || true
      if grep -q 'recorded (xsat)' "$STATE_DIR/cw_verify${CWTAG}_k${KDEC}.log"; then
        log "cross-check passed (xsat); DRAT archived at $DRAT"
      else
        log "cross-check inconclusive; certification stands at cpsat level"
      fi
    fi
  fi
  # phase 6: polish the best layout (ALWAYS) -- minimize total interaction range
  # while holding cutwidth <= UB. Same bond dimension, less entanglement spread.
  if [[ "${POLISH_TIME%.*}" -gt 0 ]]; then
    log "phase 6: polishing total interaction range at cutwidth $UB ($(_dur "$POLISH_TIME"))"
    C cw-polish --target "$UB" --time "$POLISH_TIME" --workers "$WORKERS" \
      --symmetry "$SYMMETRY" ${CWFLAGS[@]+"${CWFLAGS[@]}"}
  fi
  C cw-export ${CWFLAGS[@]+"${CWFLAGS[@]}"} \
    > "$STATE_DIR/${CLUSTER}_cw${CWTAG}_permutation.txt"
  if [[ "$CLOSED" -eq 1 ]]; then
    log "result: cutwidth c*(${CLUSTER}) = $UB (CERTIFIED); permutation in"
  else
    log "result: best-known cutwidth $UB (lower bound $LB, NOT certified); permutation in"
  fi
  log "  $STATE_DIR/${CLUSTER}_cw${CWTAG}_permutation.txt"
  exit 0
fi

# ---------------------------------------------------------------- phase 1
log "phase 1: structural bounds"
if [[ "$ORBITS" -eq 1 ]]; then
  C info --orbits
else
  C info
fi

# ---------------------------------------------------------------- phase 2
log "phase 2: heuristic upper bound ($(_dur "$TIME_HEUR"), ${PROCS} procs, seed ${SEED})"
C heuristic --time "$TIME_HEUR" --procs "$PROCS" --seed "$SEED" --stall "$STALL"

# ---------------------------------------------------------------- phase 3
log "phase 3: CP-SAT optimize pass ($(_dur "$TIME_OPT"))"
C optimize --time "$TIME_OPT" --workers "$WORKERS" --symmetry "$SYMMETRY"

# ---------------------------------------------------------------- phase 4
log "phase 4: decision ladder, ${JOBS_PER_SIDE} parallel jobs per side"
round=0
while true; do
  read -r LB UB < <(window)
  log "window: [$LB, $UB]"
  if [[ "$UB" -ge 0 && "$LB" -ge "$UB" ]]; then
    log "window closed."
    break
  fi
  round=$((round + 1))
  if [[ "$round" -gt "$MAX_ROUNDS" ]]; then
    log "MAX_ROUNDS reached without closing the window; raise TIME_PER_K."
    break
  fi
  pids=()
  # SAT side: k = UB-1, UB-2, ...   (each success lowers UB)
  for ((i = 0; i < JOBS_PER_SIDE; i++)); do
    k=$((UB - 1 - i))
    [[ "$k" -lt "$LB" ]] && continue
    log "  launching decide k=$k (SAT side)"
    C decide --k "$k" --time "$TIME_PER_K" --workers "$WORKERS" \
      --symmetry "$SYMMETRY" >"$STATE_DIR/decide_k${k}_r${round}.log" 2>&1 &
    pids+=($!)
  done
  # UNSAT side: k = LB, LB+1, ...   (each infeasibility raises LB)
  for ((i = 0; i < JOBS_PER_SIDE; i++)); do
    k=$((LB + i))
    [[ "$k" -ge $((UB - JOBS_PER_SIDE)) ]] && continue   # avoid double-booking
    log "  launching decide k=$k (UNSAT side)"
    C decide --k "$k" --time "$TIME_PER_K" --workers "$WORKERS" \
      --symmetry "$SYMMETRY" >"$STATE_DIR/decide_k${k}_r${round}.log" 2>&1 &
    pids+=($!)
  done
  if [[ "${#pids[@]}" -eq 0 ]]; then
    log "no jobs launchable in window [$LB,$UB]; stopping."
    break
  fi
  wait "${pids[@]}" || true
  # if nothing moved this round, every job timed out -> budget too small
  read -r LB2 UB2 < <(window)
  if [[ "$LB2" -eq "$LB" && "$UB2" -eq "$UB" ]]; then
    log "no progress this round (all decisions UNKNOWN). Raise TIME_PER_K"
    log "or farm more 'decide' jobs externally; state is in $STATE_DIR."
    break
  fi
done

C status

# ---------------------------------------------------------------- phase 5
read -r LB UB < <(window)
if [[ "$UB" -lt 0 ]]; then
  log "no upper bound found yet; nothing to export. Re-run to resume."
  C status
  exit 0
fi
CLOSED=0
if [[ "$LB" -ge "$UB" ]]; then
  CLOSED=1
  log "window closed: B* = $UB (certified)"
else
  log "window NOT closed: $LB <= B* <= $UB. Proceeding with the best known"
  log "ordering (bandwidth $UB); re-run to resume or farm decide jobs to tighten."
fi
KSTAR="$UB"
# phase 5: cross-check the decisive UNSAT only when the window is closed
if [[ "$CLOSED" -eq 1 ]]; then
  KDEC=$((KSTAR - 1))
  PROOF_LEVEL="$(proof_of "$KDEC")"
  if [[ "$PROOF_LEVEL" == "xsat" || "$PROOF_LEVEL" == "drat" ]]; then
    log "phase 5: decisive k=$KDEC already verified at level '$PROOF_LEVEL'; skipping"
  elif [[ "${SAT_TIME%.*}" -le 0 ]]; then
    log "phase 5: skipped (SAT_TIME=0); certification stands at cpsat level."
    log "  Permutation UNSAT instances are hard for pure CNF solvers, so the"
    log "  cross-check can far exceed the CP-SAT time; run it later with"
    log "  verify-unsat (consider FINAL_SYM=orbit for vertex-transitive graphs)."
  else
    log "phase 5: independent SAT cross-check for the decisive k=$KDEC"
    CNF="$STATE_DIR/${CLUSTER}_k${KDEC}.cnf"
    DRAT="$STATE_DIR/${CLUSTER}_k${KDEC}.drat"
    C verify-unsat --k "$KDEC" --time "$SAT_TIME" --symmetry "$FINAL_SYM" \
      --cnf-out "$CNF" --proof-out "$DRAT" \
      > "$STATE_DIR/verify_k${KDEC}.log" 2>&1 || true
    if grep -q 'proven infeasible (xsat)' "$STATE_DIR/verify_k${KDEC}.log"; then
      log "cross-check passed: CaDiCaL and Glucose both report UNSAT (proof level xsat)"
      log "DRAT proof archived at $DRAT (third-party verifiable with drat-trim)"
    elif grep -q 'instance is SAT' "$STATE_DIR/verify_k${KDEC}.log"; then
      log "ERROR: python-sat found k=$KDEC SAT — contradicts CP-SAT. Investigate!"
    else
      log "cross-check inconclusive within $(_dur "$SAT_TIME"); certification stands at"
      log "cpsat level. Re-run: $PYTHON $CERT verify-unsat --cluster $CLUSTER \\"
      log "  --state-dir $STATE_DIR --k $KDEC --time <more> --cnf-out $CNF --proof-out $DRAT"
    fi
  fi
fi
# phase 6: polish the best known ordering (ALWAYS) at its current bandwidth UB
if [[ "${POLISH_TIME%.*}" -gt 0 ]]; then
  log "phase 6: polishing total interaction range at bandwidth $KSTAR ($(_dur "$POLISH_TIME"))"
  C polish --target "$KSTAR" --time "$POLISH_TIME" --workers "$WORKERS"
fi
log "exporting ordering"
C export >"$STATE_DIR/${CLUSTER}_optimal_permutation.txt"
if [[ "$CLOSED" -eq 1 ]]; then
  log "result: B*(${CLUSTER}) = $KSTAR (CERTIFIED); permutation in"
else
  log "result: best-known bandwidth $KSTAR (lower bound $LB, NOT certified); permutation in"
fi
log "  $STATE_DIR/${CLUSTER}_optimal_permutation.txt"
C status
