# lucidrt — fine-tune + fast inference for plain-language medical text

Fine-tune a small modern LLM (**Qwen2.5-1.5B-Instruct**) with LoRA to rewrite dense
clinical / biomedical text into **plain language**, then build an **inference-optimization
harness** (KV cache, quantization, batching, profiling) and benchmark the speedups.

> ⚕️ **Not for clinical use.** This is an educational / portfolio project trained on public,
> de-identified data ([`bigbio/medparasimp`](https://huggingface.co/datasets/bigbio/medparasimp),
> author-written plain-language summaries of Cochrane abstracts). It simplifies text; it does
> not give medical advice.

## Why this project
Two signals in one repo:
1. **ML / training** — fine-tune a current LLM (LoRA/QLoRA) in PyTorch.
2. **ML systems / inference engineering** — KV caching, int8 quantization, dynamic batching,
   profiling, and a real benchmark report.

## 4-day plan
| Day | Focus | Deliverable |
|-----|-------|-------------|
| 1 | PyTorch fundamentals + data pipeline | `src/data.py`: `Dataset` + `DataLoader` producing correctly-shaped batches |
| 2 | Fine-tuning (training loop) | `src/train.py`: LoRA fine-tune, loss/perplexity, sample generations |
| 3 | Inference opt. part 1 | `src/generate.py`: naive loop vs **KV cache**; fp16 + int8 **quantization** |
| 4 | Inference opt. part 2 + polish | `src/benchmark.py`: **batching** sweep, **profiling**, plots, report; optional Gradio demo |

## How you'll work (concepts-first)
- `src/*.py` has the real structure with **`# TODO(you)`** markers where you implement the key PyTorch.
- `reference/*.py` has a working reference to check against **after** you try.
- Run it on **free Google Colab (T4 GPU)** — see below.

## Running on Colab
```python
# In a Colab cell (Runtime > Change runtime type > T4 GPU):
!git clone <your-repo-url> lucidrt
%cd lucidrt
!pip install -q -r requirements.txt
# Day 1:
!python -m src.data --inspect        # inspect the raw dataset
!python -m src.data --smoke-test     # build a DataLoader, print one batch's shapes
```

## Layout
```
lucidrt/
  src/
    data.py        # Day 1  (you implement the TODOs)
    train.py       # Day 2
    generate.py    # Day 3
    benchmark.py   # Day 4
  reference/
    data_reference.py   # Day 1 solution — peek after you try
  requirements.txt
```
