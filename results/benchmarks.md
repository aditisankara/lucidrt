# LucidRT — benchmark results

Model: **Qwen2.5-1.5B-Instruct** · Task: plain-language medical text simplification
(`lighteval/med_paragraph_simplification`). Numbers come from several GPU sessions
(Colab T4 + NUS SoC TITAN V), so latency/throughput are **directional**, not single-machine.
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

| batch | vLLM TOTAL tok/s | vLLM per-seq tok/s |
|---:|---:|---:|
| 1  | 153.5  | 153.5 |
| 8  | 929.5  | 116.2 |
| 16 | 1503.3 | 94.0  |
| 32 | 2174.2 | 67.9  |
| 64 | 2767.6 | 43.2  |

- vLLM throughput scales **153 → 2768 tok/s** (batch 1→64) and keeps climbing; per-seq degrades gracefully (153 → 43). That's PagedAttention + continuous batching (vLLM reported max concurrency 561×).
- The hand-built pipeline **saturated ~124 tok/s at batch 16** with per-seq collapsing to 7.7.
- **Honest caveat:** the hand-built numbers were on T4 / TITAN V; vLLM here is on an A100. So the raw ~12× at batch 16 mixes *engine + hardware*. The clean, hardware-independent signal is the **scaling shape** — vLLM sustains throughput under load where static batching collapses. A same-A100 rerun of the hand-built loop would isolate the engine contribution.
- Output verified coherent / on-task (quality intact).

## Still to do

- (optional) same-A100 rerun of the hand-built pipeline to isolate engine vs hardware
- (optional) **variable-length** batching · final write-up
