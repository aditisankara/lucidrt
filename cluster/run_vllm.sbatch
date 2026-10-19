#!/bin/bash
# Unattended vLLM benchmark. Submit from repo root: sbatch cluster/run_vllm.sbatch
#SBATCH --job-name=lucidrt-vllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1      # A100 (CC 8.0) — vLLM needs >= 7.5; the 'nv' TITANs incl. CC 7.0 TITAN V which vLLM can't run
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                 # host RAM — vLLM reserves CPU blocks for KV-cache swap; default mem gets OOM-killed
#SBATCH --time=01:00:00
#SBATCH --output=%j.vllm.out

bash "$SLURM_SUBMIT_DIR/cluster/run_vllm.sh"
