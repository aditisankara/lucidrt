"""
Day 1 — Data pipeline + PyTorch fundamentals.

Goal of today: understand Dataset / DataLoader / padding / attention_mask by
building a pipeline that turns the medparasimp dataset into batches of correctly
shaped tensors. No training yet (that's Day 2).

Concepts you are practicing:
  - torch.utils.data.Dataset   : defines ONE example (__len__, __getitem__)
  - torch.utils.data.DataLoader: batches + shuffles examples
  - collate_fn : stitches N variable-length examples into one rectangular tensor
                 by PADDING to the longest in the batch, and builds attention_mask
                 (1 = real token, 0 = padding) so the model ignores the padding.

YOU WRITE THE LOGIC. Each function below has a spec and a signature; the body is
yours. The runner at the bottom (_inspect / _smoke_test) is your acceptance test —
you don't need to touch it, just make it pass.

Stuck on a *concept*? Ask me and I'll explain it — I won't paste the code.

Run (in Colab):
  python -m src.data --inspect       # look at the raw data first (ALWAYS do this)
  python -m src.data --smoke-test    # build a DataLoader and print a batch's shapes

Acceptance criteria for Day 1:
  --inspect     prints the splits, column names, and one readable example
  --smoke-test  prints  input_ids shape: torch.Size([4, L])  and a decoded example
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
# You'll need these two — look up their APIs:
#   from datasets import load_dataset
#   from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # ungated; swap to meta-llama/Llama-3.2-1B-Instruct later if you like
DATASET_NAME = "bigbio/medparasimp"
MAX_LEN = 512

# The instruction we wrap every example in. Qwen is an *instruct* model, so a clear
# instruction + the input + a cue for the answer is the right shape.
PROMPT_TEMPLATE = (
    "Simplify the following medical text into plain language a patient can understand.\n\n"
    "Medical text:\n{technical}\n\nPlain language:\n"
)


def load_raw():
    """Return the raw HuggingFace dataset (a DatasetDict of splits).

    Spec:
      - load DATASET_NAME from the HuggingFace hub
      - bigbio loading scripts need trust_remote_code=True
    """
    raise NotImplementedError


def get_tokenizer():
    """Return a tokenizer for MODEL_NAME, ready for batching.

    Spec:
      - load the tokenizer for MODEL_NAME
      - causal LMs often have NO pad token; if pad_token is missing, set it to eos_token
        (otherwise the collator can't pad)
      - return the tokenizer
    """
    raise NotImplementedError


class MedSimplifyDataset(Dataset):
    """One example = one (technical, plain) pair, formatted + tokenized.

    __getitem__ should return a dict of 1-D LongTensors:
        {"input_ids": <LongTensor [L]>, "attention_mask": <LongTensor [L]>}
    for the concatenated prompt + target. (Label masking for training is Day 2.)
    """

    def __init__(self, hf_split, tokenizer, technical_key: str, plain_key: str):
        self.data = hf_split
        self.tok = tokenizer
        self.technical_key = technical_key
        self.plain_key = plain_key

    def __len__(self):
        """Spec: return how many examples are in self.data."""
        raise NotImplementedError

    def __getitem__(self, idx):
        """Spec:
          1. grab row = self.data[idx]; pull technical + plain text via the key fields
          2. build the input string: PROMPT_TEMPLATE.format(technical=...) + plain + eos
          3. tokenize it (truncation=True, max_length=MAX_LEN)
          4. return {"input_ids": <1-D LongTensor>, "attention_mask": <1-D LongTensor>}
        Think about: the tokenizer returns python lists by default — how do you get
        a 1-D tensor of shape [L] (not [1, L])?
        """
        raise NotImplementedError


@dataclass
class PadCollator:
    """Pads a list of variable-length examples into one rectangular batch.

    This is THE Day-1 concept. Try it by hand before reaching for any helper.
    """
    pad_token_id: int

    def __call__(self, batch):
        """`batch` is a list of dicts, each {"input_ids": [L_i], "attention_mask": [L_i]}
        with DIFFERENT lengths L_i.

        Spec: return {"input_ids": [B, max_len], "attention_mask": [B, max_len]} where
          - max_len is the longest input_ids IN THIS BATCH
          - shorter examples are right-padded: input_ids with self.pad_token_id,
            attention_mask with 0
          - the per-example 1-D tensors are stacked into one 2-D tensor
        Think about: why must the mask be 0 exactly where you padded?
        """
        raise NotImplementedError


def build_dataloaders(batch_size: int = 4):
    """Wire everything together: tokenizer + dataset + collator -> DataLoaders.

    Spec:
      - get the tokenizer and the raw dataset
      - set technical_key / plain_key to the REAL column names you saw from --inspect
      - set the split names (train / validation / test) to what --inspect printed
      - build a MedSimplifyDataset for train and val
      - build a PadCollator(pad_token_id=tokenizer.pad_token_id)
      - return (train_dl, val_dl, tokenizer)
        train_dl shuffles; val_dl does not; both use collate_fn=the collator
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Below is your acceptance test / runner. You shouldn't need to edit it —
# just make load_raw() and build_dataloaders() work so it passes.
# ----------------------------------------------------------------------------
def _inspect():
    raw = load_raw()
    print("Splits:", {k: len(v) for k, v in raw.items()})
    first_split = next(iter(raw.values()))
    print("Columns:", first_split.column_names)
    print("\n--- one raw example ---")
    for k, v in first_split[0].items():
        preview = str(v)
        print(f"{k}: {preview[:200]}{'...' if len(preview) > 200 else ''}")


def _smoke_test():
    train_dl, _, tok = build_dataloaders(batch_size=4)
    batch = next(iter(train_dl))
    print("batch keys:", list(batch.keys()))
    print("input_ids shape:", batch["input_ids"].shape, batch["input_ids"].dtype)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("\ndecoded example 0:\n", tok.decode(batch["input_ids"][0], skip_special_tokens=True)[:400])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    if args.inspect:
        _inspect()
    if args.smoke_test:
        _smoke_test()
    if not (args.inspect or args.smoke_test):
        print(__doc__)
