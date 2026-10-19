#!/bin/bash
# Unattended vLLM benchmark. Submit from repo root: sbatch cluster/run_vllm.sbatch
#SBATCH --job-name=lucidrt-vllm
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=%j.vllm.out

bash "$SLURM_SUBMIT_DIR/cluster/run_vllm.sh"
