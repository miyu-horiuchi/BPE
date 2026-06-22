#!/usr/bin/env python3
"""Genome tokenizer scaling table — single NT vs BPE sweep vs GPT-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bpe.corpus import corpus_to_corpus_file, genome_snippets_from_proteins, load_corpus
from bpe.report import print_table, write_genome_table
from bpe.tokenizers import DomainBPETrainer
from bpe.zipf import compute_distribution_metrics

GENOME_VOCAB_SWEEP = (16, 50, 100, 250, 500, 1000, 2000, 4000, 8000)


class SingleNucleotideTokenizer:
    name = "single_nt"

    @property
    def vocab_size(self) -> int:
        return 4

    def tokenize(self, sequence: str) -> list[str]:
        return [c for c in sequence.upper() if c in "ACGT"]

    def encode(self, sequence: str) -> list[int]:
        m = {"A": 0, "C": 1, "G": 2, "T": 3}
        return [m.get(c, 4) for c in sequence.upper()]

    def decode(self, ids: list[int]) -> str:
        inv = "ACGTN"
        return "".join(inv[i] if i < len(inv) else "N" for i in ids)


class GPT2GenomeTokenizer:
    """GPT-2 BPE applied to raw nucleotide strings (negative control)."""

    name = "gpt2_genome"

    def __init__(self) -> None:
        from transformers import GPT2TokenizerFast

        self._tok = GPT2TokenizerFast.from_pretrained("gpt2")

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    def tokenize(self, sequence: str) -> list[str]:
        return self._tok.tokenize(sequence)


def _gpt2_effective_vocab(sequences, tokenizer: GPT2GenomeTokenizer) -> int:
    from collections import Counter

    c: Counter[str] = Counter()
    for s in sequences:
        c.update(tokenizer.tokenize(s))
    return len(c)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    proteins = load_corpus(
        Path(args.data_dir),
        source=args.corpus,
        max_sequences=args.max_sequences,
    )
    windows = genome_snippets_from_proteins(
        proteins, window=args.window, stride=args.stride
    )
    print(f"Corpus: {len(windows)} genome windows ({args.window}bp)")

    corpus_path = out / "genome_corpus.txt"
    corpus_to_corpus_file(windows, corpus_path)

    vocab_sizes = [int(v) for v in args.vocab_sizes.split(",")]
    tok_dir = out / "tokenizers"

    table_rows = [
        compute_distribution_metrics(
            windows,
            SingleNucleotideTokenizer(),
            display_name="Single nucleotide (ACGT)",
            vocab_override=4,
        )
    ]

    for vs in vocab_sizes:
        print(f"  training BPE vocab={vs}...")
        tok = DomainBPETrainer(vocab_size=vs).train(
            corpus_path, name=f"genome_bpe_{vs}"
        )
        tok.save(tok_dir / f"genome_bpe_{vs}")
        actual_vocab = tok.vocab_size
        table_rows.append(
            compute_distribution_metrics(
                windows,
                tok,
                display_name=f"BPE vocab={vs}",
                vocab_override=actual_vocab,
            )
        )

    gpt2 = GPT2GenomeTokenizer()
    gpt2_vocab = _gpt2_effective_vocab(windows, gpt2)
    table_rows.append(
        compute_distribution_metrics(
            windows,
            gpt2,
            display_name="GPT-2 (English BPE)",
            vocab_override=gpt2_vocab,
        )
    )

    path = write_genome_table(table_rows, out)
    print_table(table_rows, genome=True)

    (out / "summary.json").write_text(
        json.dumps({"n_windows": len(windows), "rows": [r.to_dict() for r in table_rows]}, indent=2)
    )
    print(f"Wrote {path}")


def main() -> None:
    default = ",".join(str(v) for v in GENOME_VOCAB_SWEEP)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default="results/genome")
    p.add_argument("--corpus", default="synthetic")
    p.add_argument("--max-sequences", type=int, default=3000)
    p.add_argument("--vocab-sizes", default=default)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    run(p.parse_args())


if __name__ == "__main__":
    main()
