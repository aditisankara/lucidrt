#!/bin/bash
# vLLM capstone benchmark on the SoC cluster. SEPARATE venv (vLLM pins its own torch,
# so don't reuse the notebook env). Run inside a GPU session, or via run_vllm.sbatch.
#   srun --partition=gpu --gpus=1 --time=01:00:00 --pty bash
#   cd ~/lucidrt && bash cluster/run_vllm.sh
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="/tmp/$USER/lucidrt-vllm"
mkdir -p "$WORK" "$HOME/out"
export TMPDIR="/tmp/$USER/pip-vllm"; mkdir -p "$TMPDIR"
export HF_HOME="$WORK/hf"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap,memory.free --format=csv
# NOTE: vLLM wants compute capability >= 7.0. A TITAN V is 7.0 (borderline — may error);
# a TITAN RTX is 7.5 (safe). If it refuses to start, exit and reroll for a 7.5+ card.

# vLLM brings its own torch build — let it; do NOT pre-install torch here.
/usr/bin/python3.12 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir vllm datasets

python "$REPO/src/bench_vllm.py" | tee "$HOME/out/vllm_bench.txt"
echo "=== done -> ~/out/vllm_bench.txt  (scp it to your laptop) ==="
