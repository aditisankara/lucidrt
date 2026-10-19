#!/bin/bash
# NUS SoC cluster — build a throwaway env in /tmp and run the whole notebook on the GPU.
#
# Usage (interactive — recommended first time so you can watch it):
#   ssh -J aditisan@stujump.comp.nus.edu.sg aditisan@xlogin.comp.nus.edu.sg
#   # clone/pull the repo into HOME (small; code only):  git clone <url> ~/lucidrt
#   srun --partition=gpu --gpus=1 --time=02:00:00 --pty bash
#   cd ~/lucidrt && bash cluster/run_soc.sh
#
# Or unattended:  sbatch cluster/run_soc.sbatch   (from ~/lucidrt)
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"     # repo root — keep this in HOME (persistent)
WORK="/tmp/$USER/lucidrt"                      # heavy work on big local /tmp (WIPED per job)
mkdir -p "$WORK"
export TMPDIR="/tmp/$USER/pip"; mkdir -p "$TMPDIR"   # keep pip temp off tiny home
export HF_HOME="$WORK/hf"                       # ~3GB model download goes to /tmp, NOT home

echo "=== GPU ==="; nvidia-smi -L
echo "repo=$REPO  work=$WORK"

# --- env: system python 3.12 + venv in /tmp; cu126 torch works on ANY SoC GPU ----
/usr/bin/python3.12 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
pip install --no-cache-dir \
    "transformers>=4.44" "datasets>=2.20" accelerate peft \
    bitsandbytes evaluate rouge-score \
    jupyter nbconvert ipykernel

# OPTIONAL — AWQ int4 backend. Fragile on new torch; we already HAVE the int4-awq
# numbers (ROUGE-L 0.0969, 13.1 tok/s, 1.15GB), so this is off by default. The AWQ
# cells will just error under --allow-errors and the rest of the run continues.
# Uncomment to attempt re-running AWQ on the cluster:
# pip install --no-cache-dir gptqmodel autoawq optimum || echo "WARN: AWQ backend install failed (ok, skipping)"

python -c "import torch; print('cuda ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# --- run the whole notebook headless; outputs written in-place (repo is in HOME) ---
#   --allow-errors        : one failing cell (e.g. AWQ) won't abort the rest
#   timeout=-1            : the 60-step training + slow int8 eval exceed the 30s default
jupyter nbconvert --to notebook --execute --inplace --allow-errors \
    --ExecutePreprocessor.timeout=-1 \
    "$REPO/src/notebooks/lucidrt_data_pipeline.ipynb"

# --- snapshot the executed notebook into persistent home for scp-ing off --------
mkdir -p "$HOME/out"
cp "$REPO/src/notebooks/lucidrt_data_pipeline.ipynb" "$HOME/out/lucidrt_executed.ipynb"
echo "=== done -> ~/out/lucidrt_executed.ipynb  (scp it to your laptop) ==="
