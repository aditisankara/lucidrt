#!/bin/bash
# Hand-built pipeline benchmark on the SAME A100 as vLLM, to isolate engine vs hardware.
# Lightweight env: just torch (cu126, works on any CUDA GPU) + transformers. No vLLM.
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="/tmp/$USER/lucidrt-hb"
mkdir -p "$WORK" "$HOME/out"
export TMPDIR="/tmp/$USER/pip-hb"; mkdir -p "$TMPDIR"
export HF_HOME="$WORK/hf"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap,memory.free --format=csv

/usr/bin/python3.12 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
pip install --no-cache-dir transformers accelerate

python "$REPO/src/bench_handbuilt.py" | tee "$HOME/out/handbuilt_bench.txt"
echo "=== done -> ~/out/handbuilt_bench.txt ==="
