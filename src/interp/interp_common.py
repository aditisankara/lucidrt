"""
Shared pieces for the LucidRT fine-tune-interpretability scripts.

Everything the interp scripts need in one place: paths, the prompt template, the
eval passages, memory-safe model loading (base vs LoRA-merged), a greedy
generate() helper, and the dataset loader.

Design notes (cluster-specific):
- The 20 MB LoRA adapter is persisted in HOME (ADAPTER_DIR) so it SURVIVES jobs.
  train_adapter.py writes it once; every other script just loads it.
- The ~3 GB base model + HF cache live in /tmp (set via HF_HOME in the runner),
  which is wiped per job -> re-downloaded each run. That's fine; the expensive
  thing we avoid repeating is the 300-step TRAINING, not the download.
- Models are loaded ONE AT A TIME with low_cpu_mem_usage=True. Holding base + ft
  together is what OOM'd the free-Colab host RAM; here we load, use, free().
"""
import gc
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Repo root (this file is at <repo>/src/interp/interp_common.py) and a place for outputs.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(REPO, "results", "interp")

# Persistent adapter location. HOME survives cluster jobs; override via env if you like.
ADAPTER_DIR = os.environ.get(
    "LUCIDRT_ADAPTER_DIR", os.path.join(os.path.expanduser("~"), "lucidrt-lora")
)

PROMPT_TEMPLATE = (
    "Simplify the following medical text into plain language a patient can understand.\n\n"
    "Medical text:\n{technical}\n\nPlain language:\n"
)

# In-domain (medical) probes. Same 3 the notebook uses, so the behavioral read and the
# activation analyses later are on identical inputs.
EVAL_SAMPLES = [
    "Two trials met the inclusion criteria. Both were small, reported results inadequately, "
    "and were of low methodological quality, giving very-low certainty evidence.",
    "The intervention group showed a statistically significant reduction in systolic blood "
    "pressure relative to placebo at twelve weeks (p < 0.05).",
    "Participants received topical application weekly for six weeks; the primary endpoint was "
    "complete wound epithelialization at follow-up.",
]

_TOK = None


def tokenizer():
    """Lazily load + cache the tokenizer (pad_token defaults to eos, as in training)."""
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(MODEL)
        if _TOK.pad_token is None:
            _TOK.pad_token = _TOK.eos_token
    return _TOK


def load_base_config():
    """Model config only (num_hidden_layers, hidden_size, ...) — no weights, no GPU."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(MODEL)


def load_base():
    """The frozen base instruct model."""
    return (
        AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16, low_cpu_mem_usage=True
        )
        .to(DEVICE)
        .eval()
    )


def load_ft(adapter_dir=None, merge=True):
    """Base + LoRA adapter. merge=True folds the delta into the weights so it behaves like a
    normal model (needed for activation hooks); merge=False keeps the PeftModel wrapper
    (needed when you want to read the A/B adapter matrices directly, e.g. weight-delta)."""
    from peft import PeftModel

    adapter_dir = adapter_dir or ADAPTER_DIR
    if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        raise FileNotFoundError(
            f"No adapter at {adapter_dir}. Run train_adapter.py first "
            f"(or set LUCIDRT_ADAPTER_DIR)."
        )
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)
    m = PeftModel.from_pretrained(m, adapter_dir)
    if merge:
        m = m.merge_and_unload()
    return m.eval()


def free(*objs):
    """Drop references + reclaim GPU/host memory. Call between loading base and ft."""
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def generate(model, technical, max_new_tokens=160):
    """Greedy-simplify one passage; returns ONLY the generated text (prompt stripped)."""
    tok = tokenizer()
    enc = tok(PROMPT_TEMPLATE.format(technical=technical), return_tensors="pt").to(DEVICE)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id
    )
    gen_ids = out[0][enc["input_ids"].shape[1]:]  # tokens AFTER the prompt
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def load_med_split(split="train", n=None):
    """lighteval/med_paragraph_simplification: row = {query: technical, answer: plain}."""
    from datasets import load_dataset

    ds = load_dataset("lighteval/med_paragraph_simplification")[split]
    return ds.select(range(min(n, len(ds)))) if n else ds


# --- cheap hallucination proxy -------------------------------------------------------------
_NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers(text):
    """All numeric tokens in a string (years, counts, p-values, RR/CI, ...)."""
    return set(_NUM.findall(text))


def fabricated_numbers(source, output):
    """Numbers that appear in the OUTPUT but not the SOURCE = invented specifics.
    A blunt but honest signal for the 'emits review boilerplate with made-up stats' failure."""
    return sorted(numbers(output) - numbers(source), key=lambda s: (len(s), s))
