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

# --- dataset registry -----------------------------------------------------------------------
# Two simplification datasets, SAME task shape (technical paragraph -> plain paragraph) but very
# different target quality. This is what makes the "fix" story possible: train on each, compare.
#   med    = the TRAP. Its 'plain' answers are grade-13 Cochrane review summaries (HARDER than the
#            input), so maximizing ROUGE against them taught register-capture, not simplification.
#   scilay = the CLEAN one. paniniDot/sci_lay has genuine plain-English lay summaries of biomedical
#            abstracts, so the target IS simpler than the source (verify with diagnose_dataset.py).
DATASETS = {
    "med": {
        "hf": "lighteval/med_paragraph_simplification",
        "config": None,
        "eval_split": "validation",
        "complex_col": "query",
        "simple_col": "answer",
        "note": "TRAP: 'plain' answers are grade-13 Cochrane review summaries (harder than input)",
    },
    "scilay": {
        "hf": "paniniDot/sci_lay",
        "config": "all",                 # default config; loads all journals
        "eval_split": "validation",
        "complex_col": "technical_text",  # the abstract
        "simple_col": "plain_text",       # the plain-english lay summary
        "trust_remote_code": True,        # ships a loading script -> needs datasets<4 + this flag
        "note": "CLEAN: genuine plain-english lay summaries of biomedical abstracts",
    },
}
# Which dataset this run targets. The runner exports LUCIDRT_DATASET; defaults to the trap set so
# every existing script behaves exactly as before.
DATASET = os.environ.get("LUCIDRT_DATASET", "med")
if DATASET not in DATASETS:
    raise ValueError(f"LUCIDRT_DATASET={DATASET!r} not in {list(DATASETS)}")

# Persistent adapter location. HOME survives cluster jobs; override via env if you like.
# Each dataset gets its OWN adapter dir so the trap and clean fine-tunes never clobber each other.
# med stays at the original path (already trained & persisted); scilay -> '<base>-scilay'.
_ADAPTER_BASE = os.environ.get(
    "LUCIDRT_ADAPTER_DIR", os.path.join(os.path.expanduser("~"), "lucidrt-lora")
)
ADAPTER_DIR = _ADAPTER_BASE if DATASET == "med" else f"{_ADAPTER_BASE}-{DATASET}"


def result_path(stem):
    """Path under results/interp for an output file, suffixed by dataset so a clean-set re-run
    (e.g. patch_nnsight on scilay) doesn't overwrite the trap-set results. med keeps bare names."""
    name = stem if DATASET == "med" else f"{stem}__{DATASET}"
    return os.path.join(RESULTS, f"{name}.txt")


PROMPT_TEMPLATE = (
    "Simplify the following medical text into plain language a patient can understand.\n\n"
    "Medical text:\n{technical}\n\nPlain language:\n"
)

# In-domain (medical) probes. The first 3 are the notebook's originals (kept first so older
# results stay comparable); the rest widen the set to ~15 so the interp claims ("the register-
# switch peaks at layer 20") rest on enough diverging prompts to be more than anecdote. All are
# the same clinical-abstract register the model was fine-tuned on.
EVAL_SAMPLES = [
    "Two trials met the inclusion criteria. Both were small, reported results inadequately, "
    "and were of low methodological quality, giving very-low certainty evidence.",
    "The intervention group showed a statistically significant reduction in systolic blood "
    "pressure relative to placebo at twelve weeks (p < 0.05).",
    "Participants received topical application weekly for six weeks; the primary endpoint was "
    "complete wound epithelialization at follow-up.",
    "A total of 312 patients were randomised to receive either the study drug or matching "
    "placebo, stratified by baseline disease severity and recruiting centre.",
    "The pooled analysis demonstrated a modest but non-significant increase in adverse events "
    "in the treatment arm, driven predominantly by gastrointestinal complaints.",
    "Median progression-free survival was 9.4 months in the experimental group compared with "
    "6.1 months in controls (hazard ratio 0.68, 95% confidence interval 0.52 to 0.89).",
    "No eligible randomised controlled trials were identified; the review therefore reports on "
    "observational evidence, which is at serious risk of confounding.",
    "Adherence to the inhaled corticosteroid regimen was assessed using pharmacy refill records "
    "and was found to be suboptimal in roughly a third of participants.",
    "The meta-analysis included eight studies with substantial statistical heterogeneity "
    "(I-squared 74%), limiting the interpretability of the pooled effect estimate.",
    "Cognitive behavioural therapy delivered over ten weekly sessions produced a clinically "
    "meaningful reduction in anxiety scores that was sustained at six-month follow-up.",
    "Serum creatinine and estimated glomerular filtration rate were monitored at baseline, week "
    "four, and week twelve to detect any decline in renal function.",
    "The device failed to meet its prespecified non-inferiority margin for the primary composite "
    "endpoint of death, myocardial infarction, or target-vessel revascularisation.",
    "Breastfed infants in the supplemented cohort showed higher mean weight-for-age z-scores, "
    "although the difference did not reach statistical significance.",
    "Authors concluded that the evidence was insufficient to recommend routine screening in "
    "asymptomatic adults and called for adequately powered trials.",
    "Postoperative pain was managed with a multimodal analgesia protocol combining paracetamol, "
    "a nonsteroidal anti-inflammatory, and as-needed opioid rescue medication.",
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


def load_pairs(name=None, split="train", n=None):
    """Load any registered simplification dataset and NORMALISE it to two columns:
    'complex' (the technical source) and 'simple' (the plain-language target). This is what lets
    train/eval code run unchanged across the trap ('med') and clean ('scilay') datasets."""
    from datasets import load_dataset

    spec = DATASETS[name or DATASET]
    kw = {"trust_remote_code": True} if spec.get("trust_remote_code") else {}
    ds = (
        load_dataset(spec["hf"], spec["config"], **kw)
        if spec["config"]
        else load_dataset(spec["hf"], **kw)
    )[split]
    cc, sc = spec["complex_col"], spec["simple_col"]
    ds = ds.map(lambda ex: {"complex": ex[cc], "simple": ex[sc]})
    return ds.select(range(min(n, len(ds)))) if n else ds


def load_med_split(split="train", n=None):
    """Back-compat shim: the trap dataset with its ORIGINAL column names {query, answer}."""
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
