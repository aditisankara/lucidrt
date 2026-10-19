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
# Pin an older vLLM. The latest build hard-requires FlashInfer, which JIT-compiles with
# the node's old nvcc and fails (--compress-mode). vLLM 0.6.x uses the v0 engine + prebuilt
# FlashAttention kernels — no FlashInfer, no JIT, works on this cluster's toolkit.
# datasets dropped (its pyarrow needs numpy>=2, clashes with vllm's numpy 1.26).
# transformers pinned to vllm 0.6.3's era — latest transformers (5.x) wants torch>=2.6
# and changed the tokenizer API, which breaks this vLLM.
pip install --no-cache-dir "vllm==0.6.3.post1" "transformers==4.46.3"

python "$REPO/src/bench_vllm.py" | tee "$HOME/out/vllm_bench.txt"
echo "=== done -> ~/out/vllm_bench.txt  (scp it to your laptop) ==="
