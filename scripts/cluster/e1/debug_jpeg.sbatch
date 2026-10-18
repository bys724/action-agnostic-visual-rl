#!/bin/bash
#SBATCH --job-name=e1_dbg
#SBATCH --partition=AIP
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --output=/proj/external_group/mrg/logs/e1_dbg_%j.out
/proj/external_group/mrg/conda_envs/aavrl-train/bin/python /proj/home/mrg/bys724/action-agnostic-visual-rl/scripts/cluster/e1/debug_jpeg.py 2>&1 | grep -v -i warn
