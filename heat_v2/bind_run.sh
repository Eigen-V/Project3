#!/bin/bash
# Per-rank CPU binding wrapper:  mpirun -np P ./bind_run.sh T ./heat2d <args>
# Gives MPI rank k the T consecutive allowed CPUs [k*T, (k+1)*T) and pins its OpenMP threads inside them.
# Without this, every rank's OMP_PLACES=cores resolves to the SAME cores -> hybrid runs are oversubscribed.
# (Single-node use; local rank is taken from the launcher's environment.)
T=$1; shift
lr=${MPI_LOCALRANKID:-${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_RANK:-0}}}
if [ -n "${BIND_CPUS:-}" ]; then list=$BIND_CPUS; else list=$(taskset -pc $$ | sed 's/.*: //'); fi
cpus=(); IFS=',' read -ra parts <<< "$list"
for p in "${parts[@]}"; do
  if [[ $p == *-* ]]; then for ((c=${p%-*}; c<=${p#*-}; c++)); do cpus+=("$c"); done; else cpus+=("$p"); fi
done
start=$((lr*T))
if [ $((start+T)) -gt ${#cpus[@]} ]; then echo "bind_run: only ${#cpus[@]} CPUs for local rank $lr x $T threads; NOT binding" >&2; export OMP_NUM_THREADS=$T; exec "$@"; fi
mine=("${cpus[@]:start:T}")
mask=$(IFS=,; echo "${mine[*]}")
places=$(printf '{%s},' "${mine[@]}"); places=${places%,}
export OMP_NUM_THREADS=$T OMP_PLACES="$places" OMP_PROC_BIND=close
if [ -n "${DRY_RUN:-}" ]; then echo "local rank $lr -> cpus $mask  OMP_PLACES=$places"; exit 0; fi
exec taskset -c "$mask" "$@"
