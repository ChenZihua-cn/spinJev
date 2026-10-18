#!/bin/bash
# Verifies the SO-ARM100 assets on a compute node: load, limits, stepping, osmesa render.
# Submit:  sbatch test/job_so100.slurm
#SBATCH --job-name=so100
#SBATCH --partition=batch          # default student partition: node[1-28], MaxTime 14-00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# `conda activate` is unavailable in non-interactive shells on this cluster.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mujoco

# Compute nodes are CPU-only: OSMesa is the only usable GL backend here.
export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

cd ~/spinJev
echo "=== node: $(hostname)  job: $SLURM_JOB_ID"
python -c "import mujoco; print('mujoco', mujoco.__version__)"
python test/test_so_arm100.py --outdir "$SLURM_JOB_ID"
