#!/usr/bin/env python3
"""Downstream linear probe: are frozen features from a better-tokenized LM more
useful for a functional task?

Bits-per-residue measures compression; reviewers also want a *downstream* signal.
We build a controlled protein-family classification task: F families, each defined
by its own set of conserved motifs embedded in a shared random amino-acid
background. Crucially the family-defining motifs are *anagrams of one another*
(distinct orderings of the same fixed residue multiset) over a shared background,
so every family has a statistically identical amino-acid composition. A model
that only sees residue frequencies (single-AA, mean-pooled) is therefore near
chance; only a model whose representation captures ordered motifs / k-mers
(domain BPE) can separate the families.

Protocol (identical model size + compute per tokenizer):
  1. Train TinyGPT unsupervised (next-token) on the pooled training sequences.
  2. Freeze it; mean-pool the final hidden states into one vector per sequence.
  3. Fit a linear classifier (logistic regression) on train features.
  4. Report test accuracy. Higher = features encode family/motif structure better.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bpe.corpus import corpus_to_corpus_file
from bpe.tokenizers import (
    AminoAcidTokenizer,
    GPT2ProteinTokenizer,
    train_all_domain_bpes,
)
from experiments.train_tiny_lm import LMConfig, TinyGPT, collate_pad

AA = list("ACDEFGHIKLMNPQRSTVWY")

# Motifs are anagrams of one MOTIF_LETTERS multiset: families differ only in the
# *ordering* of the same residues, so amino-acid composition is identical across
# families and only an order-sensitive (subword) representation can separate them.
MOTIF_LETTERS = "ACDEFGHK"


def _anagram_bank(n: int, seed: int) -> list[str]:
    import itertools

    perms = ["".join(p) for p in itertools.permutations(MOTIF_LETTERS)]
    rng = random.Random(seed)
    rng.shuffle(perms)
    return perms[:n]


def make_family_corpus(
    n_families: int, per_family: int, motifs_per_family: int,
    min_len: int, max_len: int, seed: int,
) -> tuple[list[str], list[int]]:
    rng = random.Random(seed)
    bank = _anagram_bank(n_families * motifs_per_family, seed)
    fam_motifs = []
    for f in range(n_families):
        fam_motifs.append(bank[f * motifs_per_family:(f + 1) * motifs_per_family])
    seqs, labels = [], []
    for f in range(n_families):
        for _ in range(per_family):
            length = rng.randint(min_len, max_len)
            parts: list[str] = []
            while sum(len(p) for p in parts) < length:
                if rng.random() < 0.30:
                    parts.append(rng.choice(fam_motifs[f]))
                else:
                    parts.append("".join(rng.choices(AA, k=rng.randint(3, 10))))
            seqs.append("".join(parts)[:length])
            labels.append(f)
    idx = list(range(len(seqs)))
    rng.shuffle(idx)
    return [seqs[i] for i in idx], [labels[i] for i in idx]


def collate(batch):
    seqs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    padded = collate_pad(seqs)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    return padded, lengths, labels


class LabeledTok(torch.utils.data.Dataset):
    def __init__(self, seqs, labels, encode_fn, max_len):
        self.items = []
        for s, y in zip(seqs, labels):
            ids = encode_fn(s)[:max_len]
            if len(ids) >= 8:
                self.items.append((torch.tensor(ids, dtype=torch.long), y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def train_lm(model, loader, steps, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    step = 0
    while step < steps:
        for padded, _lengths, _y in loader:
            padded = padded.to(device)
            inp, tgt = padded[:, :-1], padded[:, 1:]
            logits = model(inp)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step >= steps:
                break


@torch.no_grad()
def extract_features(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats, ys = [], []
    for padded, lengths, y in loader:
        padded = padded.to(device)
        h = model.hidden(padded)  # (B, T, D)
        b, t, _ = h.shape
        pos = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
        valid = (pos < lengths.to(device).unsqueeze(1)).float().unsqueeze(-1)
        pooled = (h * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        feats.append(pooled.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


def probe_accuracy(Xtr, ytr, Xte, yte) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(Xtr), ytr)
    return float(clf.score(scaler.transform(Xte), yte))


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    seqs, labels = make_family_corpus(
        args.families, args.per_family, args.motifs_per_family,
        args.min_len, args.max_len_seq, args.seed,
    )
    split = int(len(seqs) * 0.8)
    tr_seqs, te_seqs = seqs[:split], seqs[split:]
    tr_lab, te_lab = labels[:split], labels[split:]
    print(f"Family task: {args.families} families, {len(seqs)} seqs "
          f"({len(tr_seqs)} train / {len(te_seqs)} test)")

    corpus_path = out / "corpus.txt"
    corpus_to_corpus_file(tr_seqs, corpus_path)
    bpe_vocabs = [int(v) for v in args.bpe_vocabs.split(",")]
    domain = train_all_domain_bpes(corpus_path, out / "tokenizers", bpe_vocabs)

    tokenizers = [(AminoAcidTokenizer(), "single_aa")]
    for vs in bpe_vocabs:
        tokenizers.append((domain[vs], f"domain_bpe_{vs}"))
    if not args.no_gpt2:
        try:
            tokenizers.append((GPT2ProteinTokenizer(), "gpt2_on_protein"))
        except Exception as exc:
            print(f"  (skipping GPT-2: {exc})")

    chance = 1.0 / args.families
    results = []
    for tok, name in tokenizers:
        torch.manual_seed(args.seed)
        cfg = LMConfig(d_model=args.d_model, n_heads=args.n_heads,
                       n_layers=args.n_layers, max_len=args.max_len, dropout=0.1)
        tr_ds = LabeledTok(tr_seqs, tr_lab, tok.encode, cfg.max_len)
        te_ds = LabeledTok(te_seqs, te_lab, tok.encode, cfg.max_len)
        lm_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                               drop_last=True, collate_fn=collate)
        feat_tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        feat_te = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        model = TinyGPT(tok.vocab_size, cfg).to(device)
        train_lm(model, lm_loader, args.steps, args.lr, device)
        Xtr, ytr = extract_features(model, feat_tr, device)
        Xte, yte = extract_features(model, feat_te, device)
        acc = probe_accuracy(Xtr, ytr, Xte, yte)
        results.append({
            "name": name, "vocab_size": int(tok.vocab_size),
            "probe_accuracy": round(acc, 4), "chance": round(chance, 4),
            "lift_over_chance": round(acc - chance, 4),
        })
        print(f"  {name:18s} vocab={tok.vocab_size:6d}  probe acc={acc:.3f}  "
              f"(chance={chance:.3f})")

    best = max(results, key=lambda r: r["probe_accuracy"])
    (out / "probe.json").write_text(
        json.dumps({"runs": results, "best": best["name"], "chance": chance}, indent=2)
    )
    lines = [
        "# Downstream linear probe — protein-family classification\n",
        f"Frozen mean-pooled features -> logistic regression. {args.families} "
        f"families, chance = {chance:.3f}. Identical model size + compute per tokenizer.\n",
        "| Tokenizer | Vocab | probe accuracy | lift over chance |",
        "|-----------|------:|---------------:|-----------------:|",
    ]
    for r in results:
        star = " **<- best**" if r["name"] == best["name"] else ""
        lines.append(
            f"| {r['name']} | {r['vocab_size']} | **{r['probe_accuracy']:.3f}**{star} "
            f"| {r['lift_over_chance']:.3f} |"
        )
    (out / "probe.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["name"] for r in results]
        accs = [r["probe_accuracy"] for r in results]
        colors = ["#C44E52" if n == "single_aa" else
                  "#DD8452" if "gpt2" in n else "#4C72B0" for n in names]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(names, accs, color=colors)
        ax.axvline(chance, ls="--", color="gray", label=f"chance = {chance:.2f}")
        ax.set_xlabel("test accuracy (higher = features more useful)")
        ax.set_title("Linear probe: protein-family classification")
        for i, v in enumerate(accs):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "probe.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"  (plot skipped: {exc})")

    print(f"\nBest probe: {best['name']} ({best['probe_accuracy']:.3f}). "
          f"Wrote {out/'probe.md'}, probe.json, probe.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/probe")
    p.add_argument("--families", type=int, default=6)
    p.add_argument("--per-family", type=int, default=400)
    p.add_argument("--motifs-per-family", type=int, default=4)
    p.add_argument("--min-len", type=int, default=120)
    p.add_argument("--max-len-seq", type=int, default=200)
    p.add_argument("--bpe-vocabs", default="256,1000")
    p.add_argument("--no-gpt2", action="store_true")
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--n-heads", type=int, default=3)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
