# DSA5208 — Scalable Distributed Computing for Data Science
## Project 3: Parallel 2D Heat Equation Solver

### 1. Project Overview

This project implements a **five-point explicit finite-difference solver for the two-dimensional heat equation**, supporting four execution modes:

- Serial
- Blocking MPI
- Nonblocking MPI
- Hybrid MPI + OpenMP

The project evaluates the performance and scalability of different parallel computing strategies, including:

- One-dimensional slab and two-dimensional Cartesian domain decompositions
- Blocking versus nonblocking MPI communication
- MPI rank and OpenMP thread configurations
- Two-dimensional process-grid shapes

The following instructions describe how to reproduce the **single-node benchmark using up to 16 workers** on the NUS Vanda HPC cluster.

---

### 2. Prerequisites

Before running the project, ensure that you have:

- The complete project source code (`heat_v2`)
- A valid NUS HPC account with access to Vanda
- Cisco Secure Client for VPN access when outside the campus network

#### Project Directory Structure

Keep the following files together in the project directory:

```text
heat_v2/
├── bind_run.sh
├── heat2d
├── Makefile
├── results/                
├── run_benchmarks.sh
├── run_validation.sh
└── vanda_benchmark.pbs
```

> **Note:** The `results/` directory is populated during execution. The `heat2d` executable is built from the source code using `make`.

---

### 3. Upload Project Files to Vanda

Connect to the NUS network using Cisco Secure Client if you are not on the campus network.

The following commands assume that an SSH alias named `vanda` has already been configured in `~/.ssh/config`. Otherwise, replace `vanda` with the appropriate SSH destination.

#### Step 1: Transfer the project folder

Open Terminal on your local machine:

```bash
cd ~/Desktop
scp -r heat_v2 vanda:~/
```

#### Step 2: Connect to Vanda

```bash
ssh vanda
```

#### Step 3: Verify the uploaded files

Once connected to Vanda:

```bash
cd ~/heat_v2
ls -l
```

Confirm that all project files have been transferred successfully.

---

### 4. Configure Permissions and Build Environment

#### Step 1: Set executable permissions

The shell scripts must have executable permissions.

In particular, `bind_run.sh` may otherwise cause MPICH to fail with an `execvp` permission-denied error, potentially resulting in an empty benchmark CSV.

Run:

```bash
chmod +x bind_run.sh run_benchmarks.sh run_validation.sh
```

#### Step 2: Check script syntax

```bash
bash -n bind_run.sh
bash -n run_benchmarks.sh
bash -n run_validation.sh
bash -n vanda_benchmark.pbs
```

All four commands should complete without producing error messages.

#### Step 3: Load the required modules

The PBS job script automatically loads:

```bash
module load gcc/13.1.0
module load mpich/ge/gcc/64/4.1.1
```

#### Step 4: Compile the program

The PBS job compiles the program on its allocated compute node using:

```bash
make clean && make
```

---

### 5. Recommended First Run: Smoke Test

Before running the complete benchmark, perform a short smoke test to verify that the program compiles, validates, and executes correctly.

#### Step 1: Submit the smoke test

From the project directory on Vanda:

```bash
cd ~/heat_v2

qsub -l walltime=00:15:00 \
  -v N_STRONG=2048,N_WEAK_BASE=512,STEPS=20,REPS=1 \
  vanda_benchmark.pbs
```

This submission:

- Requests a 15-minute walltime limit
- Uses a strong-scaling grid size of 2048
- Sets the weak-scaling base grid size to 512
- Runs 20 time steps per measurement
- Performs one repetition per benchmark configuration
- Executes correctness validation before performance benchmarking

The command-line parameters apply only to this submission and do not modify the original PBS script.

#### Step 2: Monitor job status

Save the job ID returned by `qsub`, for example:

```text
1234567.stdct-mgmt-02
```

Monitor the job:

```bash
watch -n 5 'qstat -u $USER'
```

Press `Ctrl+C` to stop monitoring.

#### Step 3: Check the completed job

Replace `JOB_ID` with the actual job ID:

```bash
qstat -xf JOB_ID | grep -E 'job_state|Exit_status|resources_used.walltime'
```

An exit status of `0` indicates that the PBS script completed successfully.

> **Important:** A successful PBS exit status does not independently guarantee that every benchmark configuration or numerical validation passed. Inspect the generated CSV files before proceeding.

---

### 6. Full Benchmark

After confirming that the smoke test completes successfully and records multiple worker counts without failures, submit the full benchmark.

```bash
cd ~/heat_v2
qsub vanda_benchmark.pbs
```

The PBS job executes the following stages in sequence:

1. Load the required GCC and MPICH modules and print node and CPU affinity information.
2. Compile the solver using `make clean && make`.
3. Execute `run_validation.sh` and save a job-specific copy of `results/validation.csv`.
4. Execute `run_benchmarks.sh`, covering experiments A–D.
5. Run an additional CPU affinity diagnostic using 4 MPI ranks and 2 OpenMP threads.

> **Note:** PBS may terminate jobs that exceed their requested walltime. In such cases, partial output files may remain in the `results/` directory.

---

### 7. Download Results to Your Local Machine

After the benchmark completes, transfer the generated results from Vanda to your local machine.

Open a **new Terminal window on your local machine**, outside the SSH session, and run:

```bash
scp -r vanda:~/heat_v2/results ~/Desktop/heat_v2/
```

The benchmark results will be copied to:

```text
~/Desktop/heat_v2/results/
```

You can then inspect the CSV files locally for further analysis and visualisation.

---
