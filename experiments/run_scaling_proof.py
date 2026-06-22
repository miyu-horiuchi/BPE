#!/usr/bin/env python3
"""Scaling proof: does Zipf-aligned (domain-BPE) tokenization train more
efficiently than single-AA or GPT-2 BPE at a *fixed* parameter budget?

This is the payoff experiment for the tokenization trap. The fair metric is
**bits per residue (bpr)** — total next-token NLL divided by the number of raw
amino-acid characters scored — so tokenizers with different vocab/compression
are directly comparable. A tokenizer that exposes language-like (p_median >= 1)
token statistics should reach lower bpr at the same model size and step count.

CPU-friendly by default: tiny model, short training, residue-capped corpus.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bpe.corpus import corpus_to_corpus_file, load_corpus
from bpe.tokenizers import (
    AminoAcidTokenizer,
    GPT2ProteinTokenizer,
    train_all_domain_bpes,
)
from bpe.zipf import compute_distribution_metrics
from experiments.train_tiny_lm import LMConfig, TinyGPT, collate_pad


class TokenDataset(Dataset):
    """Holds raw residue-capped sequences alongside their token ids so we can
    score NLL per *residue*, not per token."""

    def __init__(self, sequences: list[str], encode_fn, max_len: int):
        self.samples: list[tuple[list[int], int]] = []
        for seq in sequences:
            ids = encode_fn(seq)[:max_len]
            if len(ids) >= 8:
                # residues actually predicted = residues covered by ids[1:].
                # We approximate the first-token residue span as (n_res - covered)
                # but since no truncation happens (max_len >> tokens), covered
                # residues == len(seq) minus the residues in the first token.
                self.samples.append((ids, len(seq)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids, n_res = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), n_res


def collate(batch):
    seqs = [b[0] for b in batch]
    n_res = [b[1] for b in batch]
    padded = collate_pad(seqs)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    return padded, lengths, torch.tensor(n_res, dtype=torch.long)


def evaluate_bpr(model, loader, device) -> tuple[float, float, float]:
    """Return (bits_per_residue, bits_per_token, tokens_per_residue)."""
    model.eval()
    total_nll_nats = 0.0
    total_tokens = 0
    total_residues = 0
    with torch.no_grad():
        for padded, lengths, n_res in loader:
            padded = padded.to(device)
            inp = padded[:, :-1]
            tgt = padded[:, 1:]
            logits = model(inp)
            # mask padded positions: a position is valid if it is < length-1
            b, t = tgt.shape
            pos = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
            valid = pos < (lengths.to(device).unsqueeze(1) - 1)
            nll = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="none",
            ).reshape(b, t)
            total_nll_nats += float((nll * valid).sum().item())
            total_tokens += int(valid.sum().item())
            # residues scored ~= total residues minus first token of each seq.
            total_residues += int(n_res.sum().item())
    total_residues = max(total_residues, 1)
    total_tokens = max(total_tokens, 1)
    bpr = (total_nll_nats / total_residues) / math.log(2)
    bpt = (total_nll_nats / total_tokens) / math.log(2)
    tpr = total_tokens / total_residues
    return bpr, bpt, tpr


def train_and_eval(
    train_seqs: list[str],
    eval_seqs: list[str],
    tokenizer,
    display_name: str,
    cfg: LMConfig,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    train_ds = TokenDataset(train_seqs, tokenizer.encode, cfg.max_len)
    eval_ds = TokenDataset(eval_seqs, tokenizer.encode, cfg.max_len)
    if len(train_ds) == 0 or len(eval_ds) == 0:
        return {"name": display_name, "error": "empty dataset"}

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    device = torch.device("cpu")
    model = TinyGPT(tokenizer.vocab_size, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    step = 0
    curve: list[float] = []
    while step < steps:
        for padded, lengths, _ in train_loader:
            padded = padded.to(device)
            inp = padded[:, :-1]
            tgt = padded[:, 1:]
            logits = model(inp)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            curve.append(float(loss.item()))
            step += 1
            if step >= steps:
                break

    bpr, bpt, tpr = evaluate_bpr(model, eval_loader, device)
    p_median = compute_distribution_metrics(
        train_seqs, tokenizer, display_name=display_name
    ).p_median
    params = sum(p.numel() for p in model.parameters())
    embed_params = sum(
        p.numel() for n, p in model.named_parameters() if "emb" in n or "head" in n
    )
    return {
        "name": display_name,
        "vocab_size": int(tokenizer.vocab_size),
        "params": params,
        "non_embedding_params": params - embed_params,
        "p_median": round(p_median, 3),
        "tokens_per_residue": round(tpr, 4),
        "bits_per_residue": round(bpr, 4),
        "bits_per_token": round(bpt, 4),
        "final_train_loss": round(curve[-1], 4) if curve else float("nan"),
        "loss_curve": [round(x, 4) for x in curve],
        "steps": steps,
    }


def write_outputs(results: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    valid = [r for r in results if "error" not in r]
    best = min(valid, key=lambda r: r["bits_per_residue"]) if valid else None

    (out / "scaling_proof.json").write_text(
        json.dumps(
            {"runs": results, "best_by_bits_per_residue": best["name"] if best else None},
            indent=2,
        )
    )

    lines = [
        "# Scaling proof — bits per residue at fixed model size\n",
        "Same TinyGPT params + steps for every tokenizer; lower bits/residue = better.\n",
        "| Tokenizer | Vocab | p_median | tok/res | **bits/residue** | bits/token |",
        "|-----------|------:|---------:|--------:|-----------------:|-----------:|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | -- | -- | -- | ERROR | -- |")
            continue
        star = " **<- best**" if best and r["name"] == best["name"] else ""
        lines.append(
            f"| {r['name']} | {r['vocab_size']} | {r['p_median']:.2f} | "
            f"{r['tokens_per_residue']:.3f} | **{r['bits_per_residue']:.3f}**{star} | "
            f"{r['bits_per_token']:.3f} |"
        )
    (out / "scaling_proof.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for r in valid:
            c = r["loss_curve"]
            if len(c) > 20:
                w = 10
                c = [sum(c[max(0, i - w):i + 1]) / len(c[max(0, i - w):i + 1]) for i in range(len(c))]
            axes[0].plot(c, label=f"{r['name']} (vocab {r['vocab_size']})", alpha=0.85)
        axes[0].set_xlabel("training step")
        axes[0].set_ylabel("train loss (smoothed)")
        axes[0].set_title("Training efficiency by tokenizer")
        axes[0].legend(fontsize=8)

        names = [r["name"] for r in valid]
        bpr = [r["bits_per_residue"] for r in valid]
        colors = ["#C44E52" if n == "single_aa" else
                  "#DD8452" if "gpt2" in n else "#4C72B0" for n in names]
        axes[1].barh(names, bpr, color=colors)
        axes[1].set_xlabel("bits per residue (lower = better)")
        axes[1].set_title("Fixed-size LM: compression by tokenizer")
        for i, v in enumerate(bpr):
            axes[1].text(v, i, f" {v:.3f}", va="center", fontsize=8)

        fig.tight_layout()
        fig.savefig(out / "scaling_proof.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # plotting is best-effort
        print(f"  (plot skipped: {exc})")


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_corpus(Path(args.data_dir), source=args.corpus, max_sequences=args.max_seqs)
    seqs = [s[: args.residue_cap] for s in raw if len(s) >= 20]
    rng_split = int(len(seqs) * 0.9)
    train_seqs, eval_seqs = seqs[:rng_split], seqs[rng_split:]
    print(f"Corpus: {len(seqs)} seqs (cap {args.residue_cap} res) -> "
          f"{len(train_seqs)} train / {len(eval_seqs)} eval")

    corpus_path = out / "corpus.txt"
    corpus_to_corpus_file(train_seqs, corpus_path)

    bpe_vocabs = [int(v) for v in args.bpe_vocabs.split(",")]
    print(f"Training domain BPE tokenizers: {bpe_vocabs}")
    domain = train_all_domain_bpes(corpus_path, out / "tokenizers", bpe_vocabs)

    tokenizers = [(AminoAcidTokenizer(), "single_aa")]
    for vs in bpe_vocabs:
        tokenizers.append((domain[vs], f"domain_bpe_{vs}"))
    if not args.no_gpt2:
        try:
            tokenizers.append((GPT2ProteinTokenizer(), "gpt2_on_protein"))
        except Exception as exc:
            print(f"  (skipping GPT-2: {exc})")

    cfg = LMConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.max_len,
        dropout=0.1,
    )

    results = []
    for tok, name in tokenizers:
        print(f"  training tiny LM: {name} (vocab={tok.vocab_size})")
        results.append(
            train_and_eval(
                train_seqs, eval_seqs, tok, name, cfg,
                steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
            )
        )
        r = results[-1]
        if "error" not in r:
            print(f"    -> bits/residue={r['bits_per_residue']:.3f}  "
                  f"p_median={r['p_median']:.2f}  tok/res={r['tokens_per_residue']:.3f}")

    write_outputs(results, out)
    valid = [r for r in results if "error" not in r]
    if valid:
        best = min(valid, key=lambda r: r["bits_per_residue"])
        print(f"\nBest bits/residue: {best['name']} ({best['bits_per_residue']:.3f})")
    print(f"Wrote {out/'scaling_proof.md'}, scaling_proof.json, scaling_proof.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default="results/scaling")
    p.add_argument("--corpus", choices=["synthetic", "bundled", "uniprot"], default="synthetic")
    p.add_argument("--max-seqs", type=int, default=600)
    p.add_argument("--residue-cap", type=int, default=200)
    p.add_argument("--bpe-vocabs", default="256,1000")
    p.add_argument("--no-gpt2", action="store_true")
    # model (kept tiny for CPU)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--n-heads", type=int, default=3)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--max-len", type=int, default=256)
    # training
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
