#!/bin/bash
#SBATCH -J RCQwenvispdat
#SBATCH --account=simlai1
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-5:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logsRC/RC_q_TAYVISP_old_%j.out
#SBATCH --error=logsRC/RC_q_TAYVISP_old_%j.err
cd /projects/simlai1/Viberank/VibeRank/tests
set -euo pipefail
mkdir -p logsPairwiseVul

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

python tests/test_rank_centrality.py

echo "Finished: $(date)"
