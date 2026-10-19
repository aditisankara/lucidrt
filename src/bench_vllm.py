"""
vLLM throughput/latency benchmark for LucidRT — the capstone comparison.

Runs the SAME model + prompt shape + output length as the hand-built pipeline
(Qwen2.5-1.5B-Instruct, greedy, 50 new tokens) so the numbers are comparable.

Prompt is hardcoded (a real example from med_paragraph_simplification) so we don't
need `datasets` — its pyarrow pulls numpy>=2, which clashes with vLLM 0.6.x's numpy 1.26.

Hand-built reference (from the notebook, for context):
  single-seq ~11 tok/s  ·  batch-16 ~124 tok/s TOTAL (then saturates)
vLLM should match on single-seq and pull far ahead at higher concurrency
thanks to PagedAttention + continuous batching.
"""
import time
from vllm import LLM, SamplingParams

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
prompt = PROMPT_TEMPLATE.format(technical=TECHNICAL)

llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.9, max_model_len=2048)
sp = SamplingParams(temperature=0.0, max_tokens=50)   # greedy, 50 tokens

# warm-up (first call compiles / allocates the KV cache; don't time it)
llm.generate([prompt], sp, use_tqdm=False)

print("\n==================  vLLM BENCHMARK  ==================")
print(f"{'batch':>6} | {'wall(s)':>8} | {'TOTAL tok/s':>12} | {'per-seq tok/s':>13}")
print("-" * 52)
results = {}
for n in [1, 8, 16, 32, 64]:
    prompts = [prompt] * n
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.time() - t0
    total_tps = n * 50 / dt
    per_seq = 50 / dt
    results[n] = total_tps
    print(f"{n:>6} | {dt:>8.2f} | {total_tps:>12.1f} | {per_seq:>13.1f}")

print("-" * 52)
print(f"hand-built reference: single ~11 tok/s, batch-16 ~124 tok/s TOTAL")
if 16 in results:
    print(f"vLLM @ batch-16: {results[16]:.0f} tok/s TOTAL  "
          f"(~{results[16]/124:.1f}x the hand-built pipeline)")
print("======================================================\n")

print("--- sample vLLM output (first 240 chars) ---")
print(outs[0].outputs[0].text[:240])
