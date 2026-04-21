#!/bin/bash
#SBATCH -J parse-llama
#SBATCH --account=simlai1
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-3:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logsParseVulVifspdat/parse_%j.out
#SBATCH --error=logsParseVulVifspdat/parse_%j.err

set -euo pipefail
mkdir -p logsParse

echo "JobID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Started: $(date)"

cd /projects/simlai1/Viberank/VibeRank

source /projects/simlai1/Viberank/VibeRank/.venv/bin/activate

echo "Python: $(which python)"
python --version
nvidia-smi || true

python src/viberank/ParsingLocally/parse_responses.py

echo "Finished: $(date)"