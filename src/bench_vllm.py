"""
vLLM throughput/latency benchmark for LucidRT — the capstone comparison.

Runs the SAME model + prompt + output length as the hand-built pipeline
(Qwen2.5-1.5B-Instruct, greedy, 50 new tokens) so the numbers are comparable.

Hand-built reference (from the notebook, for context):
  single-seq ~11 tok/s  ·  batch-16 ~124 tok/s TOTAL (then saturates)
vLLM should match on single-seq and pull far ahead at higher concurrency
thanks to PagedAttention + continuous batching.
"""
import time
from datasets import load_dataset
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_TEMPLATE = (
    "Simplify the following medical text into plain language a patient can understand.\n\n"
    "Medical text:\n{technical}\n\nPlain language:\n"
)

# same prompt as the notebook (validation[0]) for an apples-to-apples compare
raw = load_dataset("lighteval/med_paragraph_simplification")
prompt = PROMPT_TEMPLATE.format(technical=raw["validation"][0]["query"])

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

# sanity: show one generated output so we know quality is intact
print("--- sample vLLM output (first 240 chars) ---")
print(outs[0].outputs[0].text[:240])
