#!/bin/bash
# Unattended version — submit from the repo root in HOME:  sbatch cluster/run_soc.sbatch
# Survives SSH disconnects (unlike an interactive srun session).
#SBATCH --job-name=lucidrt
#SBATCH --partition=gpu            # 'gpu' = 3h max; switch to 'gpu-long' (3 days) if it runs long
#SBATCH --gpus=1
#SBATCH --time=02:59:00            # under the gpu-partition 3h cap
#SBATCH --output=%j.lucidrt.out    # log file named by job id (in submit dir)

bash "$SLURM_SUBMIT_DIR/cluster/run_soc.sh"
