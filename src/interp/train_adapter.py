"""
Train the LoRA adapter ONCE and persist it to HOME so no later script ever retrains.

This is the notebook's Section 1 turned into a standalone, idempotent script:
  - if an adapter already exists at ADAPTER_DIR, it does nothing and exits;
  - otherwise it fine-tunes Qwen2.5-1.5B for --steps steps and saves the adapter.

Run (on a GPU node, from the repo root):  python src/interp/train_adapter.py
The adapter lands in ~/lucidrt-lora (20 MB, fits the home quota, survives jobs).
"""
import argparse
import os
import sys
from dataclasses import dataclass

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_common as C  # noqa: E402

MAX_LEN = 512


class MedSimplifyDataset(Dataset):
    """Each row -> 'prompt + plain answer + eos', so the model learns to produce the plain
    version after the 'Plain language:' cue. (At generation time we give only the prompt.)"""

    def __init__(self, split, tok):
        self.data, self.tok = split, tok

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        text = C.PROMPT_TEMPLATE.format(technical=ex["complex"]) + ex["simple"] + self.tok.eos_token
        enc = self.tok(text, truncation=True, max_length=MAX_LEN)
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
        }


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, batch):
        max_len = max(e["input_ids"].size(0) for e in batch)
        ids, masks = [], []
        for e in batch:
            pad = max_len - e["input_ids"].size(0)
            ids.append(torch.cat([e["input_ids"], torch.full((pad,), self.pad_token_id, dtype=torch.long)]))
            masks.append(torch.cat([e["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(masks)}


def make_labels(input_ids, attn):
    labels = input_ids.clone()
    labels[attn == 0] = -100  # ignore padding positions in the loss
    return labels


def train(adapter_dir, steps, batch_size, lr):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    tok = C.tokenizer()
    train_split = C.load_pairs(split="train")   # normalised to {complex, simple}; honours LUCIDRT_DATASET
    print(f"dataset: {C.DATASET} ({C.DATASETS[C.DATASET]['hf']}) | train rows: {len(train_split)}", flush=True)
    train_dl = DataLoader(
        MedSimplifyDataset(train_split, tok),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=PadCollator(tok.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(C.MODEL, dtype=torch.float16, device_map="cuda")
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention only
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # ~2.18M trainable (0.14%)

    # memory savers — these are what keep training inside a 12 GB card
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = GradScaler("cuda")
    model.train()

    step = 0
    while step < steps:
        for batch in train_dl:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            labels = make_labels(batch["input_ids"], batch["attention_mask"])
            opt.zero_grad()
            with autocast("cuda", dtype=torch.float16):
                loss = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels
                ).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 25 == 0:
                print(f"step {step:4d}/{steps} | loss {loss.item():.3f}", flush=True)
            if step >= steps:
                break

    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    mb = sum(
        os.path.getsize(os.path.join(adapter_dir, f)) for f in os.listdir(adapter_dir)
    ) / 1e6
    print(f"saved adapter -> {adapter_dir} ({mb:.1f} MB)")
    C.free(model, opt, scaler)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter-dir", default=C.ADAPTER_DIR)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--force", action="store_true", help="retrain even if an adapter exists")
    args = ap.parse_args()

    print("device:", C.DEVICE, "| adapter_dir:", args.adapter_dir, flush=True)
    exists = os.path.exists(os.path.join(args.adapter_dir, "adapter_config.json"))
    if exists and not args.force:
        print("Adapter already present — nothing to do. (Use --force to retrain.)")
        return
    if C.DEVICE != "cuda":
        sys.exit("Refusing to train on CPU — request a GPU node first.")
    train(args.adapter_dir, args.steps, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
