"""
Hand-built pipeline benchmark — SAME A100 as bench_vllm.py, for an apples-to-apples
engine-vs-engine comparison (isolates the engine from the hardware).

Identical to the vLLM run: same model, same hardcoded prompt, 50 greedy tokens,
same batch sweep. This is the manual KV-cache + static-batching loop from the notebook,
just run standalone so it lands on the same GPU as vLLM.
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_TEMPLATE = (
    "Simplify the following medical text into plain language a patient can understand.\n\n"
    "Medical text:\n{technical}\n\nPlain language:\n"
)
TECHNICAL = (
    "Two trials met the inclusion criteria. One compared 2% ketanserin ointment in "
    "polyethylene glycol (PEG) with PEG alone, used twice a day by 40 participants with "
    "arterial leg ulcers, for eight weeks or until healing, whichever was sooner. One "
    "compared topical application of blood-derived concentrated growth factor (CGF) with "
    "standard dressing (polyurethane film or foam); both applied weekly for six weeks by 61 "
    "participants with non-healing ulcers (venous, diabetic arterial, neuropathic, traumatic, "
    "or vasculitic). Both trials were small, reported results inadequately, and were of low "
    "methodological quality. Short follow-up times (six and eight weeks) meant it would be "
    "difficult to capture sufficient healing events to allow us to make comparisons between "
    "treatments. One trial demonstrated accelerated wound healing in the ketanserin group "
    "compared with the control group. We assessed this as very-low certainty evidence due to "
    "the small number of studies and arterial ulcer participants, inadequate reporting of "
    "methodology and data, and short follow-up period. There is insufficient evidence to "
    "determine whether the choice of topical agent or dressing affects the healing of "
    "arterial leg ulcers."
)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda").eval()
prompt = PROMPT_TEMPLATE.format(technical=TECHNICAL)
print("GPU:", torch.cuda.get_device_name(0))

MAX_NEW = 50

def run(batch_size):
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    input_ids = ids.repeat(batch_size, 1)            # [B, L] — B identical prompts (static batch)
    torch.cuda.synchronize(); t0 = time.time()
    cur, past = input_ids, None
    for _ in range(MAX_NEW):
        with torch.no_grad():
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(-1, keepdim=True)   # greedy
    torch.cuda.synchronize(); dt = time.time() - t0
    return dt, batch_size * MAX_NEW / dt, MAX_NEW / dt

# warm-up
run(1)

print("\n============  HAND-BUILT BENCHMARK (same A100)  ============")
print(f"{'batch':>6} | {'wall(s)':>8} | {'TOTAL tok/s':>12} | {'per-seq tok/s':>13}")
print("-" * 52)
for n in [1, 8, 16, 32, 64]:
    try:
        dt, total_tps, per_seq = run(n)
        print(f"{n:>6} | {dt:>8.2f} | {total_tps:>12.1f} | {per_seq:>13.1f}")
    except torch.cuda.OutOfMemoryError:
        print(f"{n:>6} | OOM")
        torch.cuda.empty_cache()
print("-" * 52)
print("Compare to vLLM on the SAME A100 (bench_vllm.py) to isolate engine vs hardware.")
print("===========================================================\n")
