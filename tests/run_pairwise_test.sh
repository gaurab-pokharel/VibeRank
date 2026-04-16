#!/bin/bash
#SBATCH -J pairwise-llama
#SBATCH --account=simlai1
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logsPairwise/pairwise_%j.out
#SBATCH --error=logsPairwise/pairwise_%j.err

set -euo pipefail
mkdir -p logsPairwise

echo "JobID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Started: $(date)"

# move to project root
cd /projects/simlai1/Viberank/VibeRank

# activate environment
source /projects/simlai1/Viberank/VibeRank/.venv/bin/activate

echo "Python: $(which python)"
python --version
nvidia-smi || true

python tests/test_pairwise.py

echo "Finished: $(date)"