# LucidRT — benchmark results

Model: **Qwen2.5-1.5B-Instruct** · Task: plain-language medical text simplification
(`lighteval/med_paragraph_simplification`). Numbers come from several GPU sessions on mixed
hardware (a Colab T4 and an older Volta card), so latency/throughput are **directional**, not single-machine.
Quality numbers are all from one clean same-session run.

## Fine-tuning (LoRA)

- Adapter: **20.2 MB**, only **0.14%** of params trainable (2.18M of 1.54B)
- Training: loss **1.88 → ~1.55** over 300 steps
- **Quality gain: +37% ROUGE-L** (base fp16 0.0971 → fine-tuned fp16 0.1326)

## Inference optimizations

| Optimization | Result |
|---|---|
| KV cache vs naive | **~1.9× faster**, bit-identical output (grows with longer outputs) |
| Prefill vs decode | decode = **97–99% of runtime**, **~400× more expensive per token** than prefill (memory-bound) |
| int8 quantization | **1.78 GB** (½ of fp16), but **~7× slower** decode |
| int4-AWQ quantization | **1.15 GB** (2.7× smaller than fp16's 3.10 GB), **13.1 tok/s** = **7.7× faster than int8** |
| Request batching | **~10× throughput** (11 → 124 tok/s at batch 16); per-seq 11 → 7.7 tok/s |

## Quality: separating the two effects (ROUGE-L, n=25, greedy, same session)

| Variant | ROUGE-L |
|---|---|
| base fp16 | 0.0971 |
| base int8 | 0.1059 |
| base int4-AWQ | 0.0955 |
| **fine-tuned fp16** | **0.1326** |

- **Fine-tuning effect:** +37% (0.0971 → 0.1326) — real, meaningful gain
- **Quantization effect:** flat within noise — quantization **preserved quality**

Absolute ROUGE is low because the reference answers are long and worded very differently;
the **gaps between variants** are the finding, not the absolute values.

## Headline takeaways

1. Fine-tuning gave a large quality gain; quantization gave memory/speed at ~no quality cost.
2. int4-AWQ dominated int8 (smaller **and** faster) with no quality loss; int8's only win was memory.
3. Decode is the memory-bound bottleneck — which is why the KV cache and batching pay off.

## vLLM comparison (capstone)

Ran on an **A100-40 (CC 8.0)**, vLLM 0.6.3.post1, same model + prompt + 50 greedy tokens.

Both on the **same A100-40 (CC 8.0)**, same model + prompt + 50 greedy tokens:

| batch | hand-built tok/s | vLLM tok/s | vLLM speedup |
|---:|---:|---:|---:|
| 1  | 44.8   | 153.5  | 3.4× |
| 8  | 355.2  | 929.5  | 2.6× |
| 16 | 740.6  | 1503.3 | 2.0× |
| 32 | 1249.3 | 2174.2 | 1.7× |
| 64 | 1915.1 | 2767.6 | 1.4× |

(per-seq tok/s at batch 64: hand-built 29.9, vLLM 43.2)

**Clean engine-vs-engine finding (identical hardware):**
- On the same A100, vLLM is **1.4–3.4× faster** — biggest at single-sequence (CUDA graphs + fused kernels cut per-step overhead the hand-written Python loop pays), narrowing to ~1.4× at batch 64.
- The gap **shrinks** as batch grows because this benchmark uses **uniform-length prompts arriving all at once — the best case for static batching** (no padding waste, no ragged finishes). vLLM's continuous batching + PagedAttention win *most* under realistic **ragged / variable-length / streaming** load, which this synthetic benchmark doesn't stress.
- The earlier "~12×" was inflated by hardware: the first hand-built numbers were on a T4/TITAN V (batch-1 ~11 tok/s → 44.8 on the A100). On equal hardware the engine advantage is 1.4–3.4×.
- Output verified coherent / on-task (quality intact).

## Still to do

- (optional) **variable-length / streaming** batching — the workload where vLLM's lead would widen
- final write-up
