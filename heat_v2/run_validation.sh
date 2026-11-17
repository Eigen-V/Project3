#!/bin/bash
# Correctness sweep. Writes results/validation.csv; exits non-zero if any case fails.
# Usage: bash run_validation.sh [mpirun-extra-args...]
set -u
mkdir -p results
OUT=results/validation.csv
RUN=${MPIRUN:-mpirun}
head=""; fail=0; total=0
: > "$OUT"
run() {  # mode decomp init n steps ranks threads
  local mode=$1 decomp=$2 init=$3 n=$4 steps=$5 p=$6 th=$7
  export OMP_NUM_THREADS=$th
  local line
  line=$($RUN -np "$p" ./heat2d --mode "$mode" --decomp "$decomp" --init "$init" --n "$n" --steps "$steps" --validate 2>&1)
  local rc=$?
  total=$((total+1))
  if [ -z "$head" ]; then head=$(grep '^mode,' <<<"$line" | head -1); [ -n "$head" ] && echo "$head" > "$OUT"; fi
  grep -E '^(serial|blocking|nonblocking|hybrid),' <<<"$line" >> "$OUT"
  if [ $rc -ne 0 ]; then fail=$((fail+1)); echo "FAIL rc=$rc: $mode $decomp $init n=$n steps=$steps P=$p T=$th"; echo "$line" | tail -3; fi
}
for init in gauss sine; do
  run serial slab $init 64 50 1 1
  for n in 17 64 101; do                  # odd / prime sizes -> uneven blocks
    for p in 1 2 3 4 6 8; do
      for d in slab cart; do
        for m in blocking nonblocking; do run $m $d $init $n 40 $p 1; done
      done
    done
  done
  for p in 2 4 8; do run hybrid cart $init 101 40 $((p/2)) 2; run hybrid slab $init 101 40 $((p/2)) 2; done
  run nonblocking cart $init 6 25 16 1    # more ranks than 2 rows per rank: tiny 1-2 cell blocks
  run nonblocking slab $init 8 25 8 1     # one row per rank
done
echo "validation cases: $total, failures: $fail  (details in $OUT)"
[ $fail -eq 0 ]
