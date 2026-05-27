#!/bin/bash
#SBATCH --job-name=calvin_diag_pertask
#SBATCH --partition=AIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/proj/external_group/mrg/logs/diag_pertask_%j.out
#SBATCH --error=/proj/external_group/mrg/logs/diag_pertask_%j.err

set -euo pipefail
module load miniforge3/24.11.3-2
source /opt/ibs_lib/apps/miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /proj/external_group/mrg/conda_envs/aavrl-train
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /proj/home/mrg/bys724/action-agnostic-visual-rl
"$CONDA_PREFIX/bin/python" -u scripts/eval/diagnose_calvin_per_task.py
