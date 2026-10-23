#!/bin/bash
# NUS SoC cluster — fine-tune interpretability runner.
# Builds a throwaway /tmp venv (cu126 torch works on ANY SoC GPU), trains the LoRA adapter
# ONCE into persistent HOME, then runs the requested analysis stage(s).
#
# Usage (interactive — recommended first time):
#   ssh -J aditisan@stujump.comp.nus.edu.sg aditisan@xlogin.comp.nus.edu.sg
#   srun --partition=gpu --gpus=1 --time=02:00:00 --pty bash
#   cd ~/lucidrt && bash cluster/run_interp.sh            # = train (if needed) + behavioral
#   cd ~/lucidrt && bash cluster/run_interp.sh behavioral # just re-run the eval (no retrain)
#
# Or unattended:  sbatch cluster/run_interp.sbatch [stage]
set -eo pipefail   # pipefail: a crash in `python ... | tee` must fail the job, not be hidden by tee's exit 0

STAGE="${1:-all}"                               # train | behavioral | all | patch | tlens | mechinterp
REPO="$(cd "$(dirname "$0")/.." && pwd)"        # repo root — keep in HOME (persistent)
WORK="/tmp/$USER/lucidrt-interp"                # heavy work on big local /tmp (WIPED per job)
mkdir -p "$WORK" "$HOME/out"
export TMPDIR="/tmp/$USER/pip-interp"; mkdir -p "$TMPDIR"
export HF_HOME="$WORK/hf"                        # ~3GB model download to /tmp, NOT tiny home

# The adapter is the ONE thing we persist across jobs — 20 MB, fits the home quota.
export LUCIDRT_ADAPTER_DIR="$HOME/lucidrt-lora"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,compute_cap,memory.free --format=csv
echo "repo=$REPO  work=$WORK  stage=$STAGE  adapter=$LUCIDRT_ADAPTER_DIR"

# --- env: system python 3.12 + venv in /tmp; cu126 torch for sm_70+ compatibility ---
/usr/bin/python3.12 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
pip install --no-cache-dir \
    "transformers>=4.44" "datasets>=2.20" accelerate peft textstat

python -c "import torch; print('cuda ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# --- stages --------------------------------------------------------------------------------
if [ "$STAGE" = "train" ] || [ "$STAGE" = "all" ]; then
    python "$REPO/src/interp/train_adapter.py"            # idempotent: skips if adapter exists
fi
if [ "$STAGE" = "behavioral" ] || [ "$STAGE" = "all" ]; then
    python "$REPO/src/interp/behavioral_eval.py" | tee "$HOME/out/behavioral_eval.txt"
fi

# --- mech-interp stages (need the adapter already trained; installs are per-stage so one
#     library's install trouble can't break the other) --------------------------------------
if [ "$STAGE" = "patch" ] || [ "$STAGE" = "mechinterp" ]; then
    pip install --no-cache-dir nnsight
    pip freeze | grep -iE "^(torch|transformers|nnsight|accelerate|peft)==" | tee "$HOME/out/patch_freeze.txt"
    python "$REPO/src/interp/patch_nnsight.py" | tee "$HOME/out/patch_nnsight.txt"
fi
if [ "$STAGE" = "logitlens" ] || [ "$STAGE" = "mechinterp" ]; then
    pip install --no-cache-dir nnsight   # fast nnsight-based logit lens (replaces the slow TransformerLens probe)
    python "$REPO/src/interp/logit_lens_nnsight.py" | tee "$HOME/out/logit_lens.txt"
fi
if [ "$STAGE" = "attnpatch" ] || [ "$STAGE" = "mechinterp" ]; then
    pip install --no-cache-dir nnsight   # sublayer-resolved attention-output patching (ties to the attention-only LoRA)
    python "$REPO/src/interp/attn_patch_nnsight.py" | tee "$HOME/out/attn_patch.txt"
fi
if [ "$STAGE" = "tlens" ]; then   # DEPRECATED: TransformerLens load is >10 min; use 'logitlens' instead
    pip install --no-cache-dir transformer_lens
    python "$REPO/src/interp/explore_tlens.py" | tee "$HOME/out/tlens_base.txt"
fi

echo "=== done. adapter in $LUCIDRT_ADAPTER_DIR ; results in $REPO/results/interp (+ ~/out) ==="
