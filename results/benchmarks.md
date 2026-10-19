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

## Still to do

- Benchmark against **vLLM** (capstone) · **variable-length** batching · final write-up
