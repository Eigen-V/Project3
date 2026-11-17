#!/bin/bash
# Scheduler-agnostic experiment driver (called by vanda_benchmark.pbs, also usable interactively).
#   Exp A  strong scaling : fixed N, all configs, core counts 1,2,4,...,NCORES
#   Exp B  weak scaling   : n = round(N_WEAK_BASE*sqrt(P)) so work/core is constant
#   Exp C  hybrid split   : at NCORES total cores, vary ranks x threads
#   Exp D  process grid   : at NCORES ranks, vary px x py (pure-MPI, nonblocking)
# Environment (all optional):
#   N_STRONG=4096 N_WEAK_BASE=1024 STEPS=200 REPS=3 NCORES=<auto> BIND=1 RESULTS=results
#   MPIRUN="mpirun"   (e.g. "mpirun.openmpi --oversubscribe" for local tests)
set -u
N_STRONG=${N_STRONG:-4096}; N_WEAK_BASE=${N_WEAK_BASE:-1024}; STEPS=${STEPS:-200}; REPS=${REPS:-3}
# NOTE: plain `nproc` returns $OMP_NUM_THREADS when set (PBS sets it to 1) -> use the affinity mask instead.
NCORES=${NCORES:-$(python3 -c "import os;print(len(os.sched_getaffinity(0)))")}; BIND=${BIND:-1}; RESULTS=${RESULTS:-results}; MPIRUN=${MPIRUN:-mpirun}
TAG=${TAG:-$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RESULTS"
OUT="$RESULTS/bench_${TAG}.csv"; LOG="$RESULTS/bench_${TAG}.log"; FAIL="$RESULTS/failures_${TAG}.csv"
echo 'exp,rep,cores,mode,decomp,init,n,steps,r,ranks,px,py,threads,seconds,comm_max,compute_max,checksum,halo_doubles_max,mcups,gflops,linf,rel_l2,analytic_err,valid' > "$OUT"
echo 'exp,rep,cores,mode,decomp,n,ranks,threads,px,exit_code' > "$FAIL"

# core counts: 1,2,4,... and NCORES itself
CORES=(); for ((c=1; c<=NCORES; c*=2)); do CORES+=("$c"); done
[ "${CORES[-1]}" -ne "$NCORES" ] && CORES+=("$NCORES")
echo "NCORES=$NCORES  core counts: ${CORES[*]}  N_STRONG=$N_STRONG N_WEAK_BASE=$N_WEAK_BASE STEPS=$STEPS REPS=$REPS BIND=$BIND"

# run exp rep mode decomp n ranks threads [px]
run() {
  local exp=$1 rep=$2 mode=$3 decomp=$4 n=$5 ranks=$6 th=$7 px=${8:-0}
  local cores=$((ranks*th)) tmp rc=0 line
  local args=(--n "$n" --steps "$STEPS" --mode "$mode" --decomp "$decomp")
  [ "$px" -gt 0 ] && args+=(--px "$px")
  tmp=$(mktemp)
  if [ "$BIND" = 1 ]; then $MPIRUN -np "$ranks" ./bind_run.sh "$th" ./heat2d "${args[@]}" > "$tmp" 2>&1 || rc=$?
  else OMP_NUM_THREADS=$th $MPIRUN -np "$ranks" ./heat2d "${args[@]}" > "$tmp" 2>&1 || rc=$?; fi
  { echo "=== exp=$exp rep=$rep $mode/$decomp n=$n ranks=$ranks threads=$th px=$px rc=$rc ==="; cat "$tmp"; } >> "$LOG"
  line=$(grep -E '^(serial|blocking|nonblocking|hybrid),' "$tmp" | tail -n 1)
  if [ $rc -ne 0 ] || [ "$(awk -F, '{print NF}' <<<"$line")" -ne 21 ]; then
    echo "WARN failed: $exp $mode/$decomp n=$n P=$ranks T=$th (rc=$rc)"
    echo "$exp,$rep,$cores,$mode,$decomp,$n,$ranks,$th,$px,$rc" >> "$FAIL"
  else echo "$exp,$rep,$cores,$line" >> "$OUT"; fi
  rm -f "$tmp"
}
wn() { python3 -c "import math;print(int(round($N_WEAK_BASE*math.sqrt($1))))"; }   # weak-scaling grid size

# warm-up (discarded): page-faults, frequency ramp-up, library start-up
echo "warm-up..."; for _ in 1 2; do $MPIRUN -np 1 ./heat2d --n "$N_STRONG" --steps "$STEPS" --mode serial > /dev/null 2>&1 || true; done

for rep in $(seq 1 "$REPS"); do
  echo "=== repetition $rep / $REPS ==="
  # ---- A: strong scaling
  run A "$rep" serial slab "$N_STRONG" 1 1
  for p in "${CORES[@]}"; do
    run A "$rep" blocking    slab "$N_STRONG" "$p" 1
    run A "$rep" blocking    cart "$N_STRONG" "$p" 1
    run A "$rep" nonblocking slab "$N_STRONG" "$p" 1
    run A "$rep" nonblocking cart "$N_STRONG" "$p" 1
    for t in 2 4; do [ $((p % t)) -eq 0 ] && [ "$p" -ge "$t" ] && run A "$rep" hybrid cart "$N_STRONG" $((p/t)) "$t"; done
  done
  # ---- B: weak scaling (serial at base n is the P=1 reference)
  run B "$rep" serial slab "$N_WEAK_BASE" 1 1
  for p in "${CORES[@]}"; do
    n=$(wn "$p")
    run B "$rep" blocking    slab "$n" "$p" 1
    run B "$rep" nonblocking cart "$n" "$p" 1
  done
  # ---- C: ranks x threads at fixed core count
  for ((t=1; t<=NCORES; t*=2)); do
    [ $((NCORES % t)) -eq 0 ] && run C "$rep" hybrid cart "$N_STRONG" $((NCORES/t)) "$t"
  done
  # ---- D: process-grid shape at fixed rank count
  for ((px=1; px<=NCORES; px*=2)); do
    [ $((NCORES % px)) -eq 0 ] && run D "$rep" nonblocking cart "$N_STRONG" "$NCORES" 1 "$px"
  done
done
echo "done: $(($(wc -l < "$OUT")-1)) ok, $(($(wc -l < "$FAIL")-1)) failed -> $OUT"
