#!/usr/bin/env python3
"""Protein tokenizer scaling table — single AA vs BPE sweep vs GPT-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bpe.corpus import corpus_to_corpus_file, load_corpus
from bpe.report import print_table, write_protein_table
from bpe.spectral import rank_bigram_merges
from bpe.tokenizers import (
    AminoAcidTokenizer,
    DomainBPETrainer,
    GPT2ProteinTokenizer,
    train_all_domain_bpes,
)
from bpe.zipf import compare_distributions, compute_distribution_metrics

PROTEIN_VOCAB_SWEEP = (50, 100, 250, 500, 1000, 2000, 4000, 8000)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    sequences = load_corpus(
        Path(args.data_dir),
        source=args.corpus,
        max_sequences=args.max_sequences,
    )
    print(f"Corpus: {len(sequences)} protein sequences")

    corpus_path = out / "corpus.txt"
    corpus_to_corpus_file(sequences, corpus_path)

    vocab_sizes = [int(v) for v in args.vocab_sizes.split(",")]
    print(f"Training BPE: {vocab_sizes}")
    tok_dir = out / "tokenizers"
    domain_bpes = train_all_domain_bpes(corpus_path, tok_dir, vocab_sizes)

    merges = rank_bigram_merges(sequences, top_k=32)
    pd_rows = [
        {"bigram": m.bigram, "score": m.score, "pmi": m.pmi}
        for m in merges
    ]
    import pandas as pd

    pd.DataFrame(pd_rows).to_csv(out / "spectral_merges.csv", index=False)

    # Build table rows in screenshot order
    table_rows = [
        compute_distribution_metrics(
            sequences,
            AminoAcidTokenizer(),
            display_name="Single AA",
            vocab_override=20,
        )
    ]
    for vs in vocab_sizes:
        tok = domain_bpes[vs]
        table_rows.append(
            compute_distribution_metrics(
                sequences,
                tok,
                display_name=f"BPE {vs}",
                vocab_override=vs,
            )
        )
    gpt2 = GPT2ProteinTokenizer()
    table_rows.append(
        compute_distribution_metrics(
            sequences,
            gpt2,
            display_name="GPT-2 (English BPE)",
            vocab_override=50257,
        )
    )

    path = write_protein_table(table_rows, out)
    print_table(table_rows, genome=False)

    summary = {
        "n_sequences": len(sequences),
        "best_p_median": max(table_rows, key=lambda r: r.p_median).tokenizer,
        "rows": [r.to_dict() for r in table_rows],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


def main() -> None:
    default = ",".join(str(v) for v in PROTEIN_VOCAB_SWEEP)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default="results/protein")
    p.add_argument("--corpus", choices=["synthetic", "bundled", "uniprot"], default="synthetic")
    p.add_argument("--max-sequences", type=int, default=5000)
    p.add_argument("--vocab-sizes", default=default)
    run(p.parse_args())


if __name__ == "__main__":
    main()
