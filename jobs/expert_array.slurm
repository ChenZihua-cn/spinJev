#!/bin/bash
# 阶段 1c: scripted-expert decision samples, one array task per shard, 20 tasks = 20 cores.
# Physics only -- no rendering, so MUJOCO_GL is deliberately not set (osmesa matters only for the
# image jobs in test/). Each task writes its own shard files, never a shared one.
#
# Submit from the cluster copy of the repo (see AGENTS.md for host/account and the rsync line):
#     sbatch jobs/expert_array.slurm
# Watch:
#     squeue -u <user>
# Collect (run locally, after the job ends):
#     rsync -az <user>@<host>:~/spinJev/data/<jobid>/ data/cluster/
#     .venv/bin/python scripts/collect_expert_data.py --merge --out data/cluster
#     sbatch --array=0-19 ... overrides: EPISODES_PER_TASK=20 sbatch jobs/expert_array.slurm
#SBATCH --job-name=experts
#SBATCH --partition=batch              # student partition: node[1-28], MaxTime 14-00
#SBATCH --array=0-19                   # 20 shards, 20 cores total, one core per task
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000MB                   # one MuJoCo model + a python interpreter; measured far below
#SBATCH --time=00:30:00                # measured ~1 s of work per episode per expert; 30 min is slack
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

# ---- knobs (override with `VAR=x sbatch ...`) -----------------------------------------
#: episodes per expert per shard; total samples ~= 20 * EPISODES_PER_TASK * n_experts * ~5
EPISODES_PER_TASK=${EPISODES_PER_TASK:-15}
#: which scripted experts run in this shard (comma-separated, see spinjev/experts/__init__.py)
EXPERTS=${EXPERTS:-reach,direct,escalate}
#: base seed; a trajectory is keyed by (seed, shard, episode), so shards never overlap
SEED=${SEED:-20261018}
#: shard index: the array task id
SHARD=${SLURM_ARRAY_TASK_ID:-0}
#: Output directory on the shared home (the cluster's ~40 TB /public; /tmp is node-local and dies
#: with the job). NB: this controller gives every array task its *own* $SLURM_JOB_ID (measured on
#: job 38202: task 14 reported "job 38217 task 14"), so the array id is what groups a run together.
OUT_DIR=${OUT_DIR:-$HOME/spinJev/data/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}

# ---- environment --------------------------------------------------------------------
# `conda activate` is unavailable in a non-interactive shell on this cluster: source first.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate mujoco            # python 3.11 + mujoco 3.3.4 (the ceiling on CentOS 7 / glibc 2.17)

# One core per task: numpy/MuJoCo must not each grab all 40 cores of the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TMPDIR="$OUT_DIR"          # keep any scratch on the shared disk, not the node's /tmp

cd "$HOME/spinJev"
mkdir -p "$OUT_DIR"

echo "=== node $(hostname)  job $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID:-0}  shard $SHARD"
python -c "import mujoco, numpy, sys; print('python', sys.version.split()[0], '| mujoco', mujoco.__version__, '| numpy', numpy.__version__)"
nproc

time python scripts/collect_expert_data.py \
    --episodes "$EPISODES_PER_TASK" \
    --seed "$SEED" \
    --shard "$SHARD" \
    --expert "$EXPERTS" \
    --out "$OUT_DIR"

ls -l "$OUT_DIR"/expert_*_shard"$(printf '%03d' "$SHARD")".jsonl
echo "=== done, shard $SHARD"
