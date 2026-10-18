#!/bin/bash
# SLURM template for MuJoCo jobs on the cluster (see CLUSTER.md).
# Submit:  sbatch test/job_mujoco.slurm
#SBATCH --job-name=mujoco
#SBATCH --partition=batch          # default student partition: node[1-28], MaxTime 14-00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # a node has 40 cores; lower this to fan out array jobs
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# `conda activate` is unavailable in non-interactive shells on this cluster.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mujoco

# Compute nodes are CPU-only: MuJoCo must use the software GL backend (OSMesa/llvmpipe).
# EGL is unusable here - Mesa >=22 needs memfd_create, which kernel 3.10 (CentOS 7) lacks.
export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/spinJev
echo "=== node: $(hostname)  job: $SLURM_JOB_ID"
python -c "import mujoco; print('mujoco', mujoco.__version__)"
python test/test_mujoco.py --outdir "$SLURM_JOB_ID"
