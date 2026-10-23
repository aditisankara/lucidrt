"""Fast nnsight-0.8 API probe for Qwen2 — run INTERACTIVELY in an srun session.

Why: patch_nnsight.py kept dying with a masked `UnboundLocalError`, and each sbatch retry rebuilds a
/tmp venv (minutes). This script isolates the exact nnsight-0.8 behaviour on THIS cluster's installed
versions in seconds: it tests four ways of saving/patching and prints what each does, with nnsight's
DEBUG on so a real failure shows real frames instead of a downstream UnboundLocalError.

Only loads the BASE model (no adapter needed). Run:
    python src/interp/probe_nnsight.py
"""
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_common as C  # noqa: E402

import nnsight  # noqa: E402
from nnsight import LanguageModel  # noqa: E402

print(f"nnsight {nnsight.__version__} | torch {torch.__version__}", flush=True)
try:
    nnsight.CONFIG.APP.DEBUG = True  # surface nnsight's internal frames in any traceback
except Exception as e:  # noqa: BLE001
    print("  (couldn't set DEBUG:", e, ")")

lm = LanguageModel(C.load_base(), tokenizer=C.tokenizer())
p = C.PROMPT_TEMPLATE.format(technical=C.EVAL_SAMPLES[0])


def section(name, fn):
    print(f"\n=== {name} ===", flush=True)
    try:
        fn()
        print(f"[{name}] OK", flush=True)
    except Exception:  # noqa: BLE001
        print(f"[{name}] FAILED:", flush=True)
        traceback.print_exc()


def a_structure():
    """What are layer.output and lm_head.output — type and shape?"""
    with lm.trace(p):
        l0 = lm.model.layers[0].output.save()
        lh = lm.lm_head.output.save()
    print("  layer0.output :", type(l0).__name__, getattr(l0, "shape", "(no shape)"))
    print("  lm_head.output:", type(lh).__name__, getattr(lh, "shape", "(no shape)"))
    if isinstance(l0, (tuple, list)):
        print("  -> layer output is a TUPLE; elem0:", type(l0[0]).__name__, getattr(l0[0], "shape", None))


def b_explicit_loop():
    """Save several layers via an explicit loop + append (frame-friendly)."""
    with lm.trace(p):
        outs = []
        for L in range(3):
            outs.append(lm.model.layers[L].output.save())
    print("  saved", len(outs), "shapes", [tuple(o.shape) if hasattr(o, "shape") else type(o).__name__ for o in outs])


def c_comprehension():
    """THE SUSPECT: save inside a list comprehension (its own Python scope)."""
    with lm.trace(p):
        outs = [lm.model.layers[L].output.save() for L in range(3)]
    print("  comprehension saved", len(outs), "shapes",
          [tuple(o.shape) if hasattr(o, "shape") else type(o).__name__ for o in outs])


def d_patch_write():
    """Patch: in-place slice write of the last position, then read logits."""
    with lm.trace(p):
        base_logits = lm.lm_head.output.save()
    with lm.trace(p):
        lm.model.layers[5].output[:, -1, :] = lm.model.layers[5].output[:, -1, :] * 0.0
        patched_logits = lm.lm_head.output.save()
    print("  base logits", tuple(base_logits.shape), "| patched logits", tuple(patched_logits.shape))
    print("  last-pos argmax base vs patched:",
          int(base_logits[0, -1].argmax()), int(patched_logits[0, -1].argmax()))


section("A: structure", a_structure)
section("B: explicit loop", b_explicit_loop)
section("C: comprehension", c_comprehension)
section("D: patch write", d_patch_write)
print("\nDONE probe", flush=True)
