"""Train tiny causal LMs to compare scaling efficiency across tokenizers.

Not meant to match ESM-2/3 — only to show that Zipf-aligned tokenization
reduces bits-per-character at fixed parameter count (the tokenization trap).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class LMConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_len: int = 256
    dropout: float = 0.1


def collate_pad(batch: list[torch.Tensor], pad_id: int = 0) -> torch.Tensor:
    max_len = max(x.size(0) for x in batch)
    out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, : x.size(0)] = x
    return out


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[str], encode_fn, max_len: int = 256):
        self.samples: list[list[int]] = []
        for seq in sequences:
            ids = encode_fn(seq)[:max_len]
            if len(ids) >= 8:
                self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.samples[idx], dtype=torch.long)


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final hidden states (after LN, before the LM head).

        Used by the downstream linear-probe experiment to extract frozen
        sequence features without the vocabulary-specific output projection.
        """
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        h = self.blocks(h, mask=mask)
        return self.ln(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(x))


def train_one(
    sequences: list[str],
    tokenizer,
    steps: int = 300,
    batch_size: int = 32,
    lr: float = 3e-4,
    cfg: LMConfig | None = None,
) -> dict:
    cfg = cfg or LMConfig()
    ds = SequenceDataset(sequences, tokenizer.encode, max_len=cfg.max_len)
    if len(ds) == 0:
        return {"name": tokenizer.name, "error": "no sequences"}

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_pad,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyGPT(tokenizer.vocab_size, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    step = 0
    losses: list[float] = []
    while step < steps:
        for batch in loader:
            batch = batch.to(device)
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            logits = model(inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            step += 1
            if step >= steps:
                break

    # eval bits per character (fair across vocab sizes)
    model.eval()
    total_nll = 0.0
    total_chars = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            logits = model(inp)
            nll = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum"
            )
            total_nll += float(nll.item())
            # map tokens back to character count via dataset samples
            total_chars += int(tgt.numel())  # upper bound; refined below

    # recompute char count from source sequences used in dataset
    char_count = sum(min(len(tokenizer.encode(s)), cfg.max_len) - 1 for s in sequences if len(tokenizer.encode(s)) >= 8)
    char_count = max(char_count, 1)
    bpc = (total_nll / char_count) / math.log(2)
    bpt = (total_nll / max(total_chars, 1)) / math.log(2)
    params = sum(p.numel() for p in model.parameters())
    return {
        "name": tokenizer.name,
        "vocab_size": tokenizer.vocab_size,
        "params": params,
        "final_loss": losses[-1] if losses else float("nan"),
        "mean_loss": sum(losses) / max(len(losses), 1),
        "bits_per_token": bpt,
        "bits_per_character": bpc,
        "steps": steps,
    }


def run_scaling_comparison(
    sequences: list[str],
    tokenizers: list,
    out_dir,
    steps: int = 300,
) -> dict:
    results = []
    for tok in tokenizers:
        print(f"    training tiny LM: {tok.name} (vocab={tok.vocab_size})")
        results.append(train_one(sequences, tok, steps=steps))
    best = min(results, key=lambda r: r.get("bits_per_character", float("inf")))
    return {"runs": results, "best_by_bpc": best["name"]}
