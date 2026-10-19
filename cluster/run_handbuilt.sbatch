#!/bin/bash
# Same-A100 hand-built benchmark. Submit from repo root: sbatch cluster/run_handbuilt.sbatch
#SBATCH --job-name=lucidrt-hb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1      # SAME card class as the vLLM run, for a fair engine-vs-engine compare
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=%j.handbuilt.out

bash "$SLURM_SUBMIT_DIR/cluster/run_handbuilt.sh"
