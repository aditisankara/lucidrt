#!/bin/bash
# Unattended fine-tune-interpretability run. Submit from the repo root in HOME:
#   sbatch cluster/run_interp.sbatch            # train (if needed) + behavioral
#   sbatch cluster/run_interp.sbatch behavioral # just the eval
#   sbatch cluster/run_interp.sbatch arc scilay # full clean-dataset arc (2nd arg = dataset)
# Survives SSH disconnects (unlike an interactive srun session).
#SBATCH --job-name=lucidrt-interp
#SBATCH --partition=gpu            # 'gpu' = 3h max; switch to 'gpu-long' (3 days) if it runs long
#SBATCH --gpus=1
#SBATCH --time=02:59:00            # under the gpu-partition 3h cap
#SBATCH --output=%j.lucidrt.out    # log file named by job id (in submit dir)

bash "$SLURM_SUBMIT_DIR/cluster/run_interp.sh" "${1:-all}" "${2:-}"
