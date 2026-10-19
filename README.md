# LucidRT — fine-tune + fast inference for plain-language medical text

Fine-tune a small modern LLM (**Qwen2.5-1.5B-Instruct**) with LoRA to rewrite dense
clinical abstracts into **plain language a patient can understand** — then build an
inference-optimization harness (KV cache, quantization, batching, profiling) and prove
every speedup with measured, apples-to-apples numbers, ending with a same-hardware
benchmark against **vLLM**.

> ⚕️ **Not for clinical use.** Educational / portfolio project trained on public,
> de-identified data ([`lighteval/med_paragraph_simplification`](https://huggingface.co/datasets/lighteval/med_paragraph_simplification) /
> author-written plain-language summaries of Cochrane abstracts). It simplifies text; it
> does not give medical advice.

---

## Why this project exists

It carries two signals in one repo:

1. **ML / training** — fine-tune a current LLM with LoRA in PyTorch, with a real loss curve
   and a quality metric.
2. **ML systems / inference engineering** — KV caching, quantization, dynamic batching,
   prefill/decode profiling, and a rigorous benchmark report that isolates *one variable at
   a time*.

The theme throughout: don't just make it work — **optimize each step, measure it against a
baseline, and keep the honest trade-offs** (including the "optimization" that made things
worse).

---

## Headline results

Model **Qwen2.5-1.5B-Instruct**; task: plain-language medical simplification. Training/opt
numbers come from several GPU sessions (Colab T4 + NUS SoC cluster), so treat latency as
**directional** — except the vLLM section below, which is a single-machine A100 run.

| Area | Result |
|---|---|
| **Fine-tune (LoRA)** | **+37% ROUGE-L** (0.097 → 0.133); adapter is **20.2 MB**, only **0.14%** of params trained (2.18M / 1.54B); loss 1.88 → ~1.55 over 300 steps |
| **KV cache** | **~1.9× faster** than a naive re-encode loop, **bit-identical** output (gap grows with output length) |
| **Prefill vs decode** | decode is **97–99% of runtime** and **~400× more expensive per token** than prefill — the bottleneck is memory-bound |
| **int4-AWQ** | **1.15 GB** (2.7× smaller than fp16's 3.10 GB), **7.7× faster decode than int8** |
| **int8** | half the memory of fp16, but **~7× slower** and slightly lower quality — a **poor trade-off here** |
| **Batching** | **~10× throughput** (11 → 124 tok/s at batch 16) by amortizing the weight load |
| **vLLM (same A100)** | **1.4–3.4× faster** than the hand-built pipeline — the pure engine gap |

**The two findings I'd lead with in an interview:**
- Fine-tuning bought real *quality*; quantization bought *memory/speed at ~no quality cost*.
  int4-AWQ dominated int8 (smaller **and** faster, no quality loss) — int8's only win was memory.
- Decode is memory-bound, which is *why* the KV cache and batching pay off at all. That one
  fact ties the whole project together.

Full tables: [`results/benchmarks.md`](results/benchmarks.md).

---

## The vLLM capstone — separating the engine from the hardware

The first time I compared my hand-built pipeline to vLLM, vLLM looked **~12× faster**. That
number was a lie — most of it was *hardware*: my pipeline had been timed on an old T4/TITAN V,
and vLLM on an A100. So I re-ran **both engines on the same A100-40**, same model, same
prompt, same 50 greedy tokens:

| batch | hand-built tok/s | vLLM tok/s | vLLM speedup |
|---:|---:|---:|---:|
| 1  | 44.8   | 153.5  | **3.4×** |
| 8  | 355.2  | 929.5  | **2.6×** |
| 16 | 740.6  | 1503.3 | **2.0×** |
| 32 | 1249.3 | 2174.2 | **1.7×** |
| 64 | 1915.1 | 2767.6 | **1.4×** |

On equal hardware the engine advantage is **1.4–3.4×** — biggest at single-sequence (CUDA
graphs + fused kernels erase per-step overhead my Python loop pays), narrowing as batch grows.

**Why the gap shrinks, and when vLLM would actually pull away:** this benchmark feeds
*uniform-length prompts arriving all at once* — the best case for simple static batching, with
no padding waste and no ragged finishes. vLLM's **continuous batching + PagedAttention** win
*most* under realistic **ragged / variable-length / streaming** load, which this synthetic test
deliberately doesn't stress. Knowing *when* an engine wins — not just that it does — is the
point of the exercise.

### War story: getting vLLM to run on the cluster

Most of the capstone's time went not into the benchmark but into making vLLM start *at all* on
a shared academic cluster with an old CUDA toolkit. Each fix is a lesson in the dependency
reality of modern inference:

- **Compute capability.** vLLM needs CC ≥ 7.0, and the cluster's default "nv" partition hands
  out TITAN V cards (CC **7.0** — borderline, and it errored). Had to specifically request an
  **A100-40 (CC 8.0)** via `--gres=gpu:a100-40:1`.
- **FlashInfer JIT.** The latest vLLM hard-requires FlashInfer, which JIT-compiles against the
  node's ancient `nvcc` and dies on an unknown `--compress-mode` flag. Fix: pin
  **`vllm==0.6.3.post1`**, which uses the v0 engine with *prebuilt* FlashAttention kernels — no
  JIT, no toolkit dependency.
- **numpy ABI clash.** `datasets`' pyarrow pulls `numpy>=2`, which clashes with vLLM 0.6.x's
  `numpy 1.26`. Fix: drop the dependency entirely and **hardcode the benchmark prompt** (a real
  example from the dataset).
- **transformers version.** Latest `transformers` (5.x) wants `torch>=2.6` and changed the
  tokenizer API, breaking this vLLM. Pinned **`transformers==4.46.3`**.
- **Host-RAM OOM.** vLLM reserves CPU blocks for KV-cache swap and got OOM-killed at the default
  memory allocation. Fix: **`--mem=48G`**.
- **Isolation.** vLLM pins its own torch build, so it lives in a **separate venv** from the
  notebook environment — never reuse the training env for the serving engine.

All of this is encoded in [`cluster/run_vllm.sh`](cluster/run_vllm.sh) /
[`cluster/run_vllm.sbatch`](cluster/run_vllm.sbatch) so the run is reproducible.

---

## Repo layout

```
lucidrt/
  src/
    data.py                     # data pipeline: Dataset + collator + DataLoaders
    bench_handbuilt.py          # hand-built KV-cache + static-batching benchmark
    bench_vllm.py               # vLLM benchmark (same model/prompt/length for parity)
    notebooks/
      lucidrt_data_pipeline.ipynb  # the main notebook: fine-tune → KV cache → quant → batching → eval
  cluster/
    run_soc.sh / run_soc.sbatch       # training / notebook env on the NUS SoC cluster
    run_handbuilt.sh / .sbatch        # hand-built benchmark on an A100
    run_vllm.sh / .sbatch             # vLLM benchmark on the SAME A100 (see war story)
  results/
    benchmarks.md               # full benchmark tables + takeaways
  README.md
```

---

## Running it

**The notebook** (fine-tune + all the inference optimizations) runs on free Colab (T4) or the
cluster's training env:

```bash
# cluster training/notebook env
bash cluster/run_soc.sh
```

**The same-hardware engine comparison** — both scripts target one A100 so the numbers are
comparable:

```bash
sbatch cluster/run_handbuilt.sbatch   # hand-built pipeline  -> ~/out/handbuilt_bench.txt
sbatch cluster/run_vllm.sbatch        # vLLM, separate venv  -> ~/out/vllm_bench.txt
```

---

## What I'd build next

- **Variable-length / streaming batching** — left-padding + attention masks for mixed-length
  prompts. This is the workload where vLLM's continuous batching lead would widen, and the
  obvious next experiment to make the comparison tell the whole story.
- An interpretability pass on *why* int4-AWQ beats int8 (outlier activation channels) and
  *where* the +37% fine-tune gain lives in the weights.

---

*Concepts touched: autoregressive decoding, KV cache, prefill vs. decode, memory-bound
roofline, quantization (int8 / int4-AWQ), LoRA/PEFT, request batching, PagedAttention &
continuous batching, and benchmarking discipline (CUDA sync, warm-up, one-variable-at-a-time).*
