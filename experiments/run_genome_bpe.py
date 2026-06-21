#!/usr/bin/env python3
"""Genome tokenizer extension: BPE on nucleotide windows derived from protein corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bpe.corpus import corpus_to_corpus_file, genome_snippets_from_proteins, load_corpus
from bpe.tokenizers import AminoAcidTokenizer, DomainBPETrainer, train_all_domain_bpes
from bpe.zipf import compare_zipf_profiles


class KmerGenomeTokenizer:
    """Single-nucleotide baseline for genome LMs (analogous to single-AA trap)."""

    name = "single_nt"

    def __init__(self) -> None:
        self._vocab = list("ACGTN")
        self._stoi = {c: i for i, c in enumerate(self._vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def tokenize(self, sequence: str) -> list[str]:
        return [c if c in self._stoi else "N" for c in sequence.upper()]

    def encode(self, sequence: str) -> list[int]:
        return [self._stoi.get(c, 4) for c in sequence.upper()]

    def decode(self, ids: list[int]) -> str:
        inv = {i: c for c, i in self._stoi.items()}
        return "".join(inv.get(i, "N") for i in ids)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    proteins = load_corpus(Path(args.data_dir), source=args.corpus, max_sequences=args.max_sequences)
    genome_windows = genome_snippets_from_proteins(proteins, window=args.window, stride=args.stride)
    corpus_path = out / "genome_corpus.txt"
    corpus_to_corpus_file(genome_windows, corpus_path)

    trainer = DomainBPETrainer(vocab_size=args.vocab_size)
    genome_bpe = trainer.train(corpus_path, name=f"genome_bpe_{args.vocab_size}")
    genome_bpe.save(out / "tokenizers" / f"genome_bpe_{args.vocab_size}")

    tokenizers = [KmerGenomeTokenizer(), genome_bpe]
    metrics = compare_zipf_profiles(genome_windows, tokenizers)
    rows = [m.to_dict() for m in metrics]
    (out / "genome_zipf.json").write_text(json.dumps(rows, indent=2))
    for m in metrics:
        print(f"{m.name}: alpha={m.zipf_exponent:.3f} entropy={m.entropy_bits:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default="results/genome_bpe")
    p.add_argument("--corpus", default="synthetic")
    p.add_argument("--max-sequences", type=int, default=1000)
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
